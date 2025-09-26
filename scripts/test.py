from ollm import Inference, file_get_contents
from ollm.gds_loader import GDSWeights, MoEWeightsLoader
from ollm.utils import Stats
from ollm.kvcache import KVCache
from ollm.gpt_oss import MyGptOssForCausalLM
from ollm import gpt_oss, gds_loader, llama, qwen3_next

import torch, os, time
from datetime import datetime
from transformers import AutoTokenizer, TextStreamer, DynamicCache

def inference_chat():
	#sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 100
	#sm, um, max_new_tokens = "[CHATS]:\n"+file_get_contents("./temp/chats.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to English).", 500
	sm, um, max_new_tokens = file_get_contents("./samples/45k_sample.txt"), "Analyze papers above and find 3 common similarities.", 500
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	input_ids = tokenizer.apply_chat_template(messages, tokenize=True, reasoning_effort="minimal", add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)
	text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
	with torch.no_grad():
		past_key_values = None #qwen3_next.Qwen3NextDiskCache(model.config, cache_dir="/media/mega4alik/ssd/kv_cache/", stats=stats) #KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/") #DynamicCache(offloading=True)
		print("\n\nGenerate started.", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", input_ids.shape)
		outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True, streamer=text_streamer).detach().cpu()
		answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)

#=======================================================
if 1==1:
	device = torch.device("cuda:0")
	model_dir ="/media/mega4alik/ssd2/models/qwen3-next-80B/"  #"/media/mega4alik/ssd/models/gpt-oss-20B/"
	print("loading", model_dir)
	qwen3_next.loader = MoEWeightsLoader(model_dir+"gds_export/", device="cuda:0")
	stats = Stats()
	qwen3_next.stats, gds_loader.stats = stats, stats
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", use_cache=True, low_cpu_mem_usage=True, ignore_mismatched_sizes=True, attn_implementation="flash_attention_2")
	#model.clean_layers_weights()
	model.eval()
	model.to(device)
	#model.offload_layers_to_gpu_cpu(gpu_layers_num=48, cpu_layers_num=0)
	#model.offload_layers_to_cpu(layers_num=48)
	inference_chat()