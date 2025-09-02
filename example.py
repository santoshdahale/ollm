from ollm import Inference, KVCache, file_get_contents

o = Inference("llama3-1B-chat", device="cuda:0") #llama3-1B-chat(3B, 8B) | gpt-oss-20B
o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #offload some layers to CPU for speed increase

sm, um = "You are helpful AI assistant", "List planets starting from Mercury"
#sm, um = file_get_contents("./samples/85k_sample.txt"), "What's common between these articles?"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
prompt = o.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = o.tokenizer(prompt, return_tensors="pt").to(o.device)

past_key_values = KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/", stats=o.stats)
outputs = o.model.generate(**inputs,  past_key_values=past_key_values, max_new_tokens=20).cpu()
answer = o.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)