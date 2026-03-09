# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer
from tokenizer import ChatFormat, Dialog, Message, Tokenizer
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.
        单GPU/CPU版本，无任何分布式/并行支持。
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        # ========== 设备设置（无分布式）==========
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ 使用设备: {device}")

        # seed
        torch.manual_seed(seed)

        # ========== 加载检查点（单文件）==========
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) == 1, f"期望1个检查点文件，找到{len(checkpoints)}个: {[p.name for p in checkpoints]}"
        
        ckpt_path = checkpoints[0]
        print(f"📁 加载检查点: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        SUPPORTED_ARGS = {
            'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size',
            'multiple_of', 'ffn_dim_multiplier', 'norm_eps', 'rope_theta'
        }

        filtered_params = {}
        for key in SUPPORTED_ARGS:
            if key in params:
                val = params[key]
                filtered_params[key] = int(val) if isinstance(val, (int, float)) and key != 'ffn_dim_multiplier' else float(val) if key == 'ffn_dim_multiplier' else val

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **filtered_params  
        )

        print(f"📊 使用参数: {filtered_params}")


        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words, f"模型vocab_size {model_args.vocab_size} != tokenizer {tokenizer.n_words}"

        # ========== 模型初始化==========
        print("🔨 初始化模型...")
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
                dtype = torch.bfloat16
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                dtype = torch.float16
        else:
            dtype = torch.float32

        if not model_parallel_is_initialized():
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='gloo',     
                    init_method='tcp://127.0.0.1:12355',
                    rank=0, 
                    world_size=1
                )
            initialize_model_parallel(1) 
            print("✅ Fairscale 初始化完成 (world_size=1)")


        model = Transformer(model_args).to(device)
        model.load_state_dict(checkpoint, strict=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型加载完成: {time.time() - start_time:.2f}秒")
        print(f"   参数量: {total_params/1e9:.2f}B ({total_params/1e6:.1f}M)")
        print(f"   精度: {dtype}, 设备: {device}")

        return Llama(model, tokenizer, device)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts.
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, f"batch_size {bsz} > max {params.max_batch_size}"

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        
        if logprobs:
            token_logprobs = torch.zeros((bsz, total_len), dtype=torch.float, device=self.device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id
        
        # 如果prompt已填满整个序列，计算logprobs
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)

        # 自回归生成
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # 只在非prompt位置替换token
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            
            # 检查EOS
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        
        # 后处理输出
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # 截取生成部分
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            
            # 截断到EOS
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    if logprobs and probs:
                        probs = probs[:eos_idx]
                    break
                except ValueError:
                    pass
            
            out_tokens.append(toks)
            out_logprobs.append(probs)
        
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    """
    Top-p (nucleus) sampling.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
