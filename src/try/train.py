"""
train.py — Stage 1: Text-only pretraining for YAE-LLM ด้วย HF Trainer
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen3_5TextConfig,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
)

VOCAB_SIZE = 32000
SEQ_LEN = 2048
TOKEN_BIN_PATH = "data/packed_tokens.bin"


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        assert os.path.exists(bin_path), f"ไม่เจอ {bin_path} ค่ะ ต้อง pack corpus ก่อน"
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_model() -> Qwen3_5ForCausalLM:
    num_layers = 20
    text_config = Qwen3_5TextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=640,
        intermediate_size=1728,
        num_hidden_layers=num_layers,
        num_attention_heads=10,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(num_layers)
        ],
        tie_word_embeddings=True,
    )
    return Qwen3_5ForCausalLM(text_config)


def main():
    dataset = PackedTokenDataset(TOKEN_BIN_PATH, SEQ_LEN)
    model = build_model()
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    args = TrainingArguments(
        output_dir="checkpoints/yae-llm-stage1",
        max_steps=20_000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch = 8*4=32
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_steps=400,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),  # bf16 ถ้ามี GPU, ไม่งั้น fp32 บน CPU
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,  # เก็บแค่ 3 checkpoint ล่าสุด กัน disk เต็ม
        report_to="none",  # เปลี่ยนเป็น "wandb" ถ้าอยากดู dashboard
        dataloader_num_workers=4,
        remove_unused_columns=False,  # สำคัญ! ไม่งั้น Trainer จะลบ column ที่ไม่รู้จักทิ้ง
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("checkpoints/yae-llm-stage1/final")
    print("✅ Stage 1 pretraining เสร็จแล้วค่ะ")


if __name__ == "__main__":
    main()
