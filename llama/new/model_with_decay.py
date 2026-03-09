# Copyright (c) Meta Platforms, Inc. and affiliates.
# This file implements a drop-in alternative Transformer that supports
# multiplicative attention decay applied to keys and values (useful to
# progressively reduce attention to prompt tokens during generation).

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from model import ModelArgs, RMSNorm, precompute_freqs_cis, apply_rotary_emb, repeat_kv


class AttentionWithDecay(nn.Module):
    """Attention layer that accepts an optional `key_decay` tensor and
    multiplicatively scales the key and value vectors for each key position
    before computing attention. Scaling k/v (instead of scores) is
    mathematically equivalent to scaling logits but keeps the implementation
    localized to the k/v side as requested.

    key_decay should have shape (bsz, cache_len + seqlen) and contain
    positive values. Values <1 reduce contribution of those keys, values >1
    increase it.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # key/value cache (same shape as original model)
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        key_decay: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)

        # Apply multiplicative decay to keys/values for key positions if provided.
        # key_decay expected shape: (bsz, cache_len + seqlen)
        if key_decay is not None:
            # ensure device and dtype match keys/values
            key_decay = key_decay.to(keys.device).type_as(keys)
            # reshape to (bsz, 1, cache_len + seqlen, 1) for broadcasting
            kd = key_decay.unsqueeze(1).unsqueeze(3)
            keys = keys * kd
            values = values * kd

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlockWithDecay(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = AttentionWithDecay(args)
        self.feed_forward = nn.Module()
        # reuse FeedForward shape as in original implementation
        hidden_dim = int(2 * args.dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.feed_forward = nn.Sequential(
            ColumnParallelLinear(args.dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x),
            nn.SiLU(),
            RowParallelLinear(hidden_dim, args.dim, bias=False, input_is_parallel=True, init_method=lambda x: x),
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        key_decay: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, key_decay=key_decay)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TransformerWithDecay(nn.Module):
    """Drop-in style Transformer that mirrors original `Transformer` but
    uses the AttentionWithDecay layer. It exposes `forward(tokens, start_pos, key_decay)`.
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlockWithDecay(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, key_decay: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            layer_key_decay = key_decay
            # For per-layer customization we could adjust key_decay differently here.
            h = layer(h, start_pos, freqs_cis, mask, key_decay=layer_key_decay)
        h = self.norm(h)
        output = self.output(h).float()
        return output
