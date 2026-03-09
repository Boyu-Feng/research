import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import wandb

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="fengboyu-the-chinese-university-of-hong-kong",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

class Training:

    def __init__(
        self,
        llama,
        train_last_layers=2,
        future_k=5,
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

        hidden_size = self.model.params.dim
        vocab_size = self.model.params.vocab_size

        # 用于预测 future token
        self.future_head = nn.Linear(hidden_size, vocab_size).to(self.device)

        self._freeze_parameters(train_last_layers)

    def _freeze_parameters(self, train_last_layers):

        print("freeze Transformer parameters")

        for p in self.model.parameters():
            p.requires_grad = False

        n_layers = len(self.model.layers)

        print(f"training {train_last_layers} layer Transformer")

        for i in range(n_layers - train_last_layers, n_layers):
            for p in self.model.layers[i].parameters():
                p.requires_grad = True

        print("training lm_head")

        for p in self.model.output.parameters():
            p.requires_grad = True

        print("training HiddenStateDecomposer")

        for p in self.decomposer.parameters():
            p.requires_grad = True

    def build_optimizer(self, lr):

        params = []

        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)

        for p in self.decomposer.parameters():
            if p.requires_grad:
                params.append(p)

        params += list(self.future_head.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr)

        return optimizer

    def load_dataset(self):

        dataset = load_dataset("gsm8k", "main")

        train_set = dataset["train"]

        print("GSM8K train size:", len(train_set))

        return train_set

    def compute_loss(self, hidden, targets):

        # hidden: [B, T, D]

        local, global_state = self.decomposer(hidden)

        B, T, D = hidden.size()

        # ===== 1. local loss (predict next token) =====

        local_logits = self.model.output(local)

        local_logits = local_logits[:, :-1, :]
        local_targets = targets[:, 1:]

        local_loss = F.cross_entropy(
            local_logits.reshape(-1, local_logits.size(-1)),
            local_targets.reshape(-1),
        )

        # ===== 2. future loss (predict future hidden mean) =====

        if T <= 1:

            future_loss = torch.tensor(0.0, device=hidden.device)

        else:

            # ---- compute suffix mean of hidden ----

            rev_hidden = torch.flip(hidden, dims=[1])
            rev_cumsum = torch.cumsum(rev_hidden, dim=1)
            suffix_sum = torch.flip(rev_cumsum, dims=[1])

            future_sum = suffix_sum[:, 1:, :]
            future_len = torch.arange(T - 1, 0, -1, device=hidden.device).view(1, -1, 1)

            future_mean = future_sum / future_len

            # stop gradient
            future_target = future_mean.detach()

            # prediction from global state
            future_pred = self.future_head(global_state[:, :-1, :])

            # cosine loss
            future_loss = 1 - F.cosine_similarity(
                future_pred,
                future_target,
                dim=-1
            )

            future_loss = future_loss.mean()

        # ===== 3. orthogonal loss =====

        local_norm = F.normalize(local, dim=-1)
        global_norm = F.normalize(global_state, dim=-1)

        dot = torch.sum(local_norm * global_norm, dim=-1)

        orth_loss = torch.mean(dot ** 2)

        # ===== total loss =====

        total_loss = (
            local_loss
            + self.future_weight * future_loss
            + self.orth_weight * orth_loss
        )

        return total_loss, local_loss, future_loss, orth_loss


    def train(
        self,
        epochs=1,
        lr=1e-4,
        max_length=512,
    ):

        dataset = self.load_dataset()

        optimizer = self.build_optimizer(lr)

        # ===== wandb init =====
        wandb.init(
            project="future-llm",
            config={
                "learning_rate": lr,
                "epochs": epochs,
                "max_length": max_length,
            }
        )

        self.model.train()
        self.decomposer.train()
        self.future_head.train()

        global_step = 0

        for epoch in range(epochs):

            print(f"\n===== Epoch {epoch} =====")

            total_loss = 0

            for sample in tqdm(dataset):

                question = sample["question"]
                answer = sample["answer"]

                text = question + "\n" + answer

                tokens = self.tokenizer.encode(text)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]

                if len(tokens) < 2:
                    continue

                input_ids = torch.tensor(
                    tokens[:-1],
                    device=self.device
                ).unsqueeze(0)

                target_ids = torch.tensor(
                    tokens[1:],
                    device=self.device
                ).unsqueeze(0)

                logits, hidden = self.model(
                    input_ids,
                    start_pos=0
                )

                loss, local_loss, future_loss, orth_loss = self.compute_loss(
                    hidden,
                    target_ids
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # ===== wandb logging =====
                wandb.log({
                    "step_loss": loss.item(),
                    "local_loss": local_loss.item(),
                    "future_loss": future_loss.item(),
                    "orth_loss": orth_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

                global_step += 1

            avg_loss = total_loss / len(dataset)

            print("Epoch Loss:", avg_loss)

            # ===== epoch log =====
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_loss
            })

        print("\nTraining Finished")

        wandb.finish()