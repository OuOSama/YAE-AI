from .model import model
from .tokenizer import YaeTokenizer

# ===================================================
#                   Tokenizer
# ===================================================
tokenizer = YaeTokenizer()

training_data = [
    "สวัสดีค่ะซามะ ยินดีต้อนรับสู่การสร้างโมเดล 3D และ VFX",
    "Qwen3ForCausalLM 200M parameters training pipeline test.",
    "เขียนโค้ด วาดภาพ สร้างโมเดล ทำแอนิเมชัน และสร้างเกมด้วย Unreal Engine",
]

print("🚀 กำลัง Train Vocab เข้า Class...")
new_tokenizer = tokenizer.train_new_from_iterator(training_data, vocab_size=32000)

new_tokenizer.save_pretrained("./my_qwen_tokenizer")
print("✨ เรียบร้อย! นำไป Encode/Decode หรือใช้กับ Qwen3 ได้ทันทีค่ะ")


# ===================================================
#                   Prepare Data
# ===================================================



# ===================================================
#                   Model
# ===================================================
print(model)


# ===================================================
#                   Traning
# ===================================================
