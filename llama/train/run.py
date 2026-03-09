from new_model import Llama
from training import Training


llama = Llama.build(
    ckpt_dir="ckpt",
    tokenizer_path="tokenizer.model",
    max_seq_len=1024,
    max_batch_size=1,
)

trainer = Training(
    llama,
    train_last_layers=2,
    future_k=5,
)

trainer.train(
    epochs=1,
    lr=1e-4,
)