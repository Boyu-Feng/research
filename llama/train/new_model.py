import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer
from separation import HiddenStateDecomposer
from tokenizer import ChatFormat, Tokenizer

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)


class Llama:

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        seed: int = 1,
    ):

        assert os.path.isdir(ckpt_dir)
        assert os.path.isfile(tokenizer_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        torch.manual_seed(seed)

        start_time = time.time()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) == 1

        ckpt_path = checkpoints[0]
        print(f"加载 checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)

        assert model_args.vocab_size == tokenizer.n_words

        if not model_parallel_is_initialized():

            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="gloo",
                    init_method="tcp://127.0.0.1:12355",
                    rank=0,
                    world_size=1,
                )

            initialize_model_parallel(1)

        print("初始化 Transformer")

        model = Transformer(model_args).to(device)

        model.load_state_dict(checkpoint, strict=False)

        model.eval()

        print(f"模型加载完成 {time.time()-start_time:.2f}s")

        return Llama(model, tokenizer, device)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: str):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.formatter = ChatFormat(tokenizer)

        hidden_size = model.params.dim

        self.decomposer = HiddenStateDecomposer(hidden_size).to(device)

        self.model.eval()
        self.decomposer.eval()

    @torch.inference_mode()
    def generate_with_decomposition(
        self,
        prompt: str,
        max_gen_len: int = 50,
        temperature: float = 0.0,
    ):

        prompt_tokens = self.tokenizer.encode(prompt)

        tokens = torch.tensor(prompt_tokens, device=self.device).unsqueeze(0)

        start_pos = 0

        generated_tokens = []

        hidden_list = []
        local_list = []
        global_list = []
        combined_list = []

        for step in range(max_gen_len):

            logits, h = self.model(tokens, start_pos)

            last_hidden = h[:, -1, :]

            local_state, global_state = self.decomposer(last_hidden)

            combined = torch.cat([local_state, global_state], dim=-1)

            hidden_list.append(last_hidden.cpu())
            local_list.append(local_state.cpu())
            global_list.append(global_state.cpu())
            combined_list.append(combined.cpu())

            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()

            generated_tokens.append(next_token_id)

            tokens = next_token

            start_pos += 1

        hidden = torch.cat(hidden_list, dim=0)
        local = torch.cat(local_list, dim=0)
        global_state = torch.cat(global_list, dim=0)
        combined = torch.cat(combined_list, dim=0)

        return {
            "tokens": generated_tokens,
            "text": self.tokenizer.decode(generated_tokens),
            "hidden": hidden,
            "local": local,
            "global": global_state,
            "combined": combined,
        }