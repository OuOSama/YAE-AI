import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Check CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using Local Model
model_id = "./model/LLM"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 💡 NEW: Manually set pad_token_id to eos_token_id to silence the warning
# Llama models don't have a pad_token by default, so we use the eos_token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    dtype=torch.float16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "Your name is Yae from genshin impact"},
    {"role": "user", "content": "what is your name?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
