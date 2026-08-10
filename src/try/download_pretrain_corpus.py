"""
download_pretrain_corpus.py — ดึง raw text สำหรับ Stage 1 pretraining
ผสม FineWeb-Edu (อังกฤษ) + Thai Wikipedia (ไทย)

Usage:
    uv run ./src/download_pretrain_corpus.py
"""

import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/pretrain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ปรับจำนวน document ตามขนาด corpus ที่ต้องการ
# เป้าหมาย ~6B tokens รวม (ตาม Chinchilla ratio สำหรับโมเดล 300M)
FINEWEB_DOCS = 3_000_000  # อังกฤษ (ประมาณ ~2-3B tokens)
WIKI_TH_DOCS = None  # None = เอาทั้งหมด (Thai Wikipedia มีจำกัดอยู่แล้ว ไม่ใหญ่มาก)


def download_fineweb_edu():
    print("กำลังโหลด FineWeb-Edu (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    out_path = os.path.join(OUTPUT_DIR, "fineweb_edu_en.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(ds, total=FINEWEB_DOCS)):
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")  # เว้นบรรทัดคั่นระหว่าง document
            if i + 1 >= FINEWEB_DOCS:
                break
    print(f"✅ เสร็จแล้ว: {out_path}")


def download_thai_wikipedia():
    print("กำลังโหลด Thai Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
    out_path = os.path.join(OUTPUT_DIR, "wikipedia_th.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        limit = WIKI_TH_DOCS or len(ds)
        for i, example in enumerate(tqdm(ds, total=limit)):
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")
            if WIKI_TH_DOCS and i + 1 >= WIKI_TH_DOCS:
                break
    print(f"✅ เสร็จแล้ว: {out_path}")


if __name__ == "__main__":
    download_thai_wikipedia()  # เล็กกว่า ทำก่อน เช็คว่า pipeline โอเค
    download_fineweb_edu()  # ใหญ่กว่า ใช้เวลานาน
