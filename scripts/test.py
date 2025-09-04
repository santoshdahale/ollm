from ollm import Inference, KVCache, file_get_contents
from ollm.gds_loader import GDSWeights
from ollm.gpt_oss import MyGptOssForCausalLM
from ollm.utils import Stats
from ollm import gpt_oss, gds_loader, llama

import torch, os
from datetime import datetime
from transformers import AutoTokenizer, TextStreamer

def inference_chat():
	#sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 20
	sm, um, max_new_tokens = "[CHATS]:\n"+file_get_contents("./temp/chats.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to english).", 500
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]	
	inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
	text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
	with torch.no_grad():
		past_key_values = KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/", stats=stats)
		print("\n\nGenerate started.", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", inputs.input_ids.shape)
		outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True, streamer=text_streamer).detach().cpu()
		answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)


#=======================================================
if 1==1:
	device = torch.device("cuda:0")
	model_dir = "/media/mega4alik/ssd/models/llama3-8B-chat/"
	print("loading", model_dir)
	llama.loader = GDSWeights(model_dir+"gds_export/", device="cuda:0")
	stats = Stats()
	llama.stats, gds_loader.stats = stats, stats
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True, attn_implementation="flash_attention_2")
	model.clean_layers_weights()
	model.eval()
	model.to(device)
	#model.offload_layers_to_cpu(layers_num=1)
	inference_chat()

elif 2==0:
	from ollm import Inference, KVCache, file_get_contents
	o = Inference("llama3-1B-chat", device="cuda:0") #llama3-1B-chat(3B, 8B) | gpt-oss-20B
	o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
	#o.offload_layers_to_cpu(layers_num=2) #offload some layers to CPU for speed increase
	model, tokenizer, device = o.model, o.tokenizer, o.device
	inference_chat()