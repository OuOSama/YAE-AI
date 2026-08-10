"""
pack_data.py — เอา corpus .txt ไป tokenize แล้ว pack เป็น binary token stream
สำหรับใช้กับ PackedTokenDataset ใน train.py

Usage:
    uv run ./src/pack_data.py
"""

import glob
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZER_PATH = "./yae_tokenizer"
CORPUS_FILES = glob.glob("data/*.txt")
OUTPUT_PATH = "data/packed_tokens.bin"
CHUNK_READ_SIZE = 50_000  # จำนวนบรรทัดที่อ่าน+tokenize ต่อรอบ กัน RAM บวม

EOS_TOKEN = "<|endoftext|>"


def main():
    assert len(CORPUS_FILES) > 0, "ไม่เจอไฟล์ corpus ใน data/ ค่ะ"
    print(f"เจอไฟล์ corpus {len(CORPUS_FILES)} ไฟล์")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    assert tokenizer is not None, f"โหลด tokenizer จาก {TOKENIZER_PATH} ไม่สำเร็จ"
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    print(f"eos_token_id = {eos_id}")

    # เขียนแบบ streaming ต่อท้ายไฟล์ binary ไปเรื่อยๆ ไม่โหลด token ทั้งหมดเข้า RAM ทีเดียว
    with open(OUTPUT_PATH, "wb") as out_f:
        total_tokens = 0

        for file_path in CORPUS_FILES:
            print(f"กำลัง tokenize: {file_path}")
            with open(file_path, encoding="utf-8") as f:
                buffer = []
                for line in tqdm(f):
                    buffer.append(line)
                    if len(buffer) >= CHUNK_READ_SIZE:
                        total_tokens += _process_chunk(buffer, tokenizer, eos_id, out_f)
                        buffer = []

                # เศษที่เหลือจากไฟล์นี้
                if buffer:
                    total_tokens += _process_chunk(buffer, tokenizer, eos_id, out_f)

    print(f"✅ Pack เสร็จแล้วค่ะ รวม {total_tokens:,} tokens → {OUTPUT_PATH}")
    print(f"   ขนาดไฟล์: {total_tokens * 4 / 1e9:.2f} GB (uint32)")


def _process_chunk(lines, tokenizer, eos_id, out_f) -> int:
    """Tokenize บรรทัดที่สะสมมา แล้วเขียน token ids (uint32) ต่อท้ายไฟล์ binary"""
    text = "".join(lines)
    if not text.strip():
        return 0

    # encode แบบไม่ใส่ special token อัตโนมัติ เพราะเราจัดการ eos เอง
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids.append(eos_id)  # คั่นแต่ละ chunk ด้วย eos เพื่อไม่ให้ document ปนกัน

    arr = np.array(ids, dtype=np.uint32)
    arr.tofile(out_f)
    return len(arr)


if __name__ == "__main__":
    main()
