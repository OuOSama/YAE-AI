from transformers.models.qwen2 import Qwen2Tokenizer


class YaeTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
