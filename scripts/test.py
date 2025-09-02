from ollm import Inference, KVCache, file_get_contents
from ollm.gds_loader import GDSWeights
from ollm.gpt_oss import MyGptOssForCausalLM
from ollm.utils import Stats
from ollm import gpt_oss, gds_loader

import torch, os
from datetime import datetime
from transformers import AutoTokenizer

def inference_chat():
	sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 10
	#sm, um, max_new_tokens = file_get_contents("./samples/2k_sample.txt"), "What's common between these article?", 20
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		past_key_values = KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/", stats=stats)
		print("\n\nGenerate started.", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", inputs.input_ids.shape)
		outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True).detach().cpu()
		answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)

#=======================================================
device = torch.device("cuda:0")
model_dir = "/media/mega4alik/ssd/models/gpt-oss-20B/"
print("loading", model_dir)
gpt_oss.loader = GDSWeights(model_dir+"gds_export/", device="cuda:0")
stats = Stats()
gpt_oss.stats = stats
gds_loader.stats = stats
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
model.eval()
model.to(device)

model.offload_layers_to_cpu(layers_num=12)
inference_chat()