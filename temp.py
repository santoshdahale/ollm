import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
	model_id = "meta-llama/Llama-3.2-1B-Instruct"  # adjust
	

	# Generate
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	sm, um = "You are helpful AI assistant", "List planets starting from Mercury"
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, truncation=True, max_length=4000, return_tensors="pt").to("cuda")
	outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False).detach().cpu()
	answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
	print(answer)

# huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./llama3-1B
