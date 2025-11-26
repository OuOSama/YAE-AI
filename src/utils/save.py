from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"
save_folder = "./model/LLM"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# save model
tokenizer.save_pretrained(save_folder)
model.save_pretrained(save_folder)
