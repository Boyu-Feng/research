import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


class Training:

    def __init__(
        self,
        llama,
        train_last_layers=2,
        future_k=8,
        future_weight=0.3,
        orth_weight=0.1,
    ):

        self.llama = llama
        self.model = llama.model
        self.decomposer = llama.decomposer
        self.tokenizer = llama.tokenizer
        self.device = llama.device

        self.future_k = future_k
        self.future_weight = future_weight
        self.orth_weight = orth_weight

        self._freeze_parameters(train_last_layers)


    # ============================
    # 冻结参数
    # ============================

    def _freeze_parameters(self, train_last_layers):

        print("Freezing Transformer parameters")

        for p in self.model.parameters():
            p.requires_grad = False

        n_layers = len(self.model.layers)

        print(f"Training last {train_last_layers} layers")

        for i in range(n_layers - train_last_layers, n_layers):
            for p in self.model.layers[i].parameters():
                p.requires_grad = True

        print("Training LM head")

        for p in self.model.output.parameters():
            p.requires_grad = True

        print("Training decomposer")

        for p in self.decomposer.parameters():
            p.requires_grad = True


    # ============================
    # optimizer
    # ============================

    def build_optimizer(self, lr):

        params = []

        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)

        for p in self.decomposer.parameters():
            if p.requires_grad:
                params.append(p)

        optimizer = torch.optim.AdamW(params, lr=lr)

        return optimizer


    # ============================
    # dataset
    # ============================

    def load_dataset(self):

        dataset = load_dataset("gsm8k", "main")
        train_set = dataset["train"]

        print("GSM8K train size:", len(train_set))

        return train_set


    # ============================
    # future hidden segmentation
    # ============================

    def segment_mean(self, hidden, K):

        """
        hidden: [B, T, D]
        return: [B, K, D]
        """

        B, T, D = hidden.shape

        seg_len = max(T // K, 1)

        reps = []

        for k in range(K):

            start = k * seg_len
            end = (k + 1) * seg_len if k < K - 1 else T

            seg = hidden[:, start:end, :]

            rep = seg.mean(dim=1)

            reps.append(rep)

        reps = torch.stack(reps, dim=1)

        return reps


    # ============================
    # MMD
    # ============================

    def compute_mmd(self, x, y):

        """
        x: [N, D]
        y: [M, D]
        """

        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        rx = (x * x).sum(dim=1).unsqueeze(1)
        ry = (y * y).sum(dim=1).unsqueeze(1)

        Kxx = torch.exp(-(rx - 2*xx + rx.t()) / x.size(1))
        Kyy = torch.exp(-(ry - 2*yy + ry.t()) / y.size(1))
        Kxy = torch.exp(-(rx - 2*xy + ry.t()) / x.size(1))

        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

        return mmd


    # ============================
    # future planning loss
    # ============================

    def compute_future_loss(self, hidden, global_state):

        """
        hidden: [B,T,D]
        global_state: [B,T,D]
        """

        B, T, D = hidden.shape

        hidden_detach = hidden.detach()

        future_segments = self.segment_mean(
            hidden_detach,
            self.future_k
        )  # [B,K,D]

        future_loss = 0

        for b in range(B):

            g = global_state[b]      # [T,D]
            f = future_segments[b]   # [K,D]

            future_loss += self.compute_mmd(g, f)

        future_loss = future_loss / B

        return future_loss


    # ============================
    # orthogonal loss
    # ============================

    def compute_orth_loss(self, local, global_state):

        local_norm = F.normalize(local, dim=-1)
        global_norm = F.normalize(global_state, dim=-1)

        dot = torch.sum(local_norm * global_norm, dim=-1)

        orth_loss = torch.mean(dot ** 2)

        return orth_loss


    # ============================
    # total loss
    # ============================

    def compute_loss(self, hidden, targets):

        """
        hidden: [B,T,D]
        """

        local, global_state = self.decomposer(hidden)

        B, T, D = hidden.shape

        # ======================
        # local LM loss
        # ======================

        local_logits = self.model.output(local)

        local_logits = local_logits[:, :-1, :]
        local_targets = targets[:, 1:]

        local_loss = F.cross_entropy(
            local_logits.reshape(-1, local_logits.size(-1)),
            local_targets.reshape(-1),
        )

        # ======================
        # future planning loss
        # ======================

        future_loss = self.compute_future_loss(
            hidden,
            global_state
        )

        # ======================
        # orthogonal loss
        # ======================

        orth_loss = self.compute_orth_loss(
            local,
            global_state
        )

        # ======================
        # total
        # ======================

        total_loss = (
            local_loss
            + self.future_weight * future_loss
            + self.orth_weight * orth_loss
        )

        return total_loss, local_loss, future_loss, orth_loss