import sys
print(sys.path)
from llama.train.new_model import Llama
from llama.train.training import Training

llama = Llama.build(
    ckpt_dir="llama/checkpoints/Meta-Llama-3.1-8B-Instruct/original",
    tokenizer_path="llama/checkpoints/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model",
    max_seq_len=1024,
    max_batch_size=1,
)

trainer = Training(
    llama,
    train_last_layers=2,
    future_k=5,
)

trainer.train(
    trainer,
    epochs=1,
    lr=1e-4,
)