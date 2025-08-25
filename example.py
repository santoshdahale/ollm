from ollm import Inference, KVCache, file_get_contents

q = Inference("llama3-1B-chat", device="cuda:0")
q.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)

sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 10
#sm, um, max_new_tokens = file_get_contents("./temp/85k_sample.txt"), "What's common between these articles?", 20
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
prompt = q.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = q.tokenizer(prompt, return_tensors="pt").to(q.device)

past_key_values = KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/", stats=q.stats) #HQQQuantizedCache
outputs = q.model.generate(**inputs,  past_key_values=past_key_values, max_new_tokens=20).cpu()
answer = q.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
print(answer); exit()

#with torch.no_grad():

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #cpu | meta, dtype=should be bfloat16
model.clean_layers_weights()
model.save_pretrained("./models/llama3-1B") #saving model without layers weights
tokenizer.save_pretrained("./models/llama3-1B"); exit()

