import os
import json
import glob
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

corpus_files = glob.glob("data/*.txt")
assert len(corpus_files) > 0, "ไม่เจอไฟล์ corpus ใน data/ นะคะ เช็ค path อีกที"
print(f"เจอไฟล์ corpus ทั้งหมด {len(corpus_files)} ไฟล์")

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
VOCAB_SIZE = 32000

tokenizer = Tokenizer(
    BPE(
        unk_token=None,
        continuing_subword_prefix="",
        end_of_word_suffix="",
        byte_fallback=False,
    )
)
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(
            Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ]
)
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<image>",
        "<video>",
    ],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)
tokenizer.train(files=corpus_files, trainer=trainer)
print("Train BPE เสร็จแล้วค่ะ")

os.makedirs("yae_tokenizer_raw", exist_ok=True)
tokenizer.model.save("yae_tokenizer_raw")

with open("yae_tokenizer_raw/vocab.json", encoding="utf-8") as f:
    trained_vocab = json.load(f)

with open("yae_tokenizer_raw/merges.txt", encoding="utf-8") as f:
    lines = f.readlines()

trained_merges = [tuple(line.split()) for line in lines[1:] if line.strip()]
print(f"vocab size จริง: {len(trained_vocab)}")
print(f"merges จำนวน: {len(trained_merges)}")

qwen_tokenizer = Qwen3_5Tokenizer(
    vocab="yae_tokenizer_raw/vocab.json",
    merges="yae_tokenizer_raw/merges.txt",
    unk_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    additional_special_tokens=[
        "<|im_start|>",
        "<|im_end|>",
        "<image>",
        "<video>",
    ],  
)
qwen_tokenizer.save_pretrained("./yae_tokenizer")
print("save เสร็จแล้วที่ ./yae_tokenizer")

test_text = "<|im_start|>user\nสวัสดีค่ะ<|im_end|>"
tokens = qwen_tokenizer.tokenize(test_text)
print("tokens:", tokens)

im_start_id = qwen_tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_start|> id: {im_start_id} | <|im_end|> id: {im_end_id}")

assert "<|im_start|>" in tokens, "❌ special token โดนตัดแยก เช็ค special_tokens อีกที"
assert "<|im_end|>" in tokens, "❌ special token โดนตัดแยก เช็ค special_tokens อีกที"
print("✅ special token ไม่โดนตัดแยก พร้อมใช้งานจริงแล้วค่ะ")
