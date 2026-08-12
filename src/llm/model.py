from transformers import Qwen3Config, Qwen3ForCausalLM


class YaeConfig(Qwen3Config): 
    pass

class YaeForCausalLM(Qwen3ForCausalLM):
    pass

config = YaeConfig(
    vocab_size=32000,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
    max_position_embeddings=32768,
    tie_word_embeddings=True,
)

model = YaeForCausalLM(config)

total_params = sum(p.numel() for p in model.parameters())

print(f"Parameters Count : {total_params:,} ({total_params / 1e6:.2f} M)")

print("\n--- Model Size in Memory ---")
print(f"FP32 (32-bit) : {total_params * 4 / (1024**2):.2f} MB ({total_params * 4 / (1024**3):.4f} GB)")
print(f"FP16 (16-bit) : {total_params * 2 / (1024**2):.2f} MB ({total_params * 2 / (1024**3):.4f} GB)")
print(f"INT8 (8-bit)  : {total_params * 1 / (1024**2):.2f} MB ({total_params * 1 / (1024**3):.4f} GB)")
print(f"INT4 (4-bit)  : {total_params * 0.5 / (1024**2):.2f} MB ({total_params * 0.5 / (1024**3):.4f} GB)")
