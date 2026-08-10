from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)  # สร้างถ้ายังไม่มี, ไม่ error ถ้ามีอยู่แล้ว

# โหลด Tulu 3 SFT mix
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")


def format_chatml(example, system_prompt=None):
    """แปลง conversation list เป็น ChatML string เดียว"""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    for turn in example["messages"]:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(parts)


# แปลงทั้ง dataset เป็น text พร้อม tokenize
with open("data/sft_chatml.txt", "w", encoding="utf-8") as f:
    for example in ds:
        formatted = format_chatml(example)
        f.write(formatted + "\n\n")  # เว้นบรรทัดคั่นระหว่าง conversation
