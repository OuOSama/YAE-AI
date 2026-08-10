from transformers import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Qwen3_5ForConditionalGeneration,
)

NUM_LAYERS = 20
text_config = Qwen3_5TextConfig(
    vocab_size=248320,
    hidden_size=640,
    intermediate_size=1728,
    num_hidden_layers=NUM_LAYERS,
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
        for i in range(NUM_LAYERS)
    ],
    tie_word_embeddings=True,
)

vision_config = Qwen3_5VisionConfig(
    hidden_size=320,
    out_hidden_size=640,
    depth=8,
    num_heads=8,
)

config = Qwen3_5Config(
    text_config=text_config,
    vision_config=vision_config,
    tie_word_embeddings=True,
)

if __name__ == "__main__":
    model = Qwen3_5ForConditionalGeneration(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"{total / 1e6:.1f}M params")  # → 301.1M
