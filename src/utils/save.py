from transformers import AutoModelForCausalLM, AutoTokenizer

llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
llm_save_folder = "./model/LLM"

emb_id = "Qwen/Qwen3-Embedding-0.6B"
emb_save_folder = "./model/EMB"
# ===========================
#  SAVE LLM Model
# ===========================
tokenizer = AutoTokenizer.from_pretrained(llm_id)
model = AutoModelForCausalLM.from_pretrained(llm_id)

tokenizer.save_pretrained(llm_save_folder)
model.save_pretrained(llm_save_folder)

# ===========================
#  SAVE EMB Model
# ===========================
tokenizer = AutoTokenizer.from_pretrained(emb_id)
model = AutoModelForCausalLM.from_pretrained(emb_id)

tokenizer.save_pretrained(emb_save_folder)
model.save_pretrained(emb_save_folder)
