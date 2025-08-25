import os, requests, zipfile
import torch
from transformers import AutoTokenizer
from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights
from . import llama

class Inference:
	def __init__(self, model_id, device="cuda:0"):
		self.model_id = model_id
		self.device = torch.device(device)
		self.stats = Stats()

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"llama3-1B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
			"llama3-8B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip"
		}
		url = urls[self.model_id]
		
		# Extract filename from URL
		filename = url.split("/")[-1]
		zip_path = os.path.join(models_dir, filename)

		# Download the file
		print(f"Downloading {url} ...")
		response = requests.get(url, stream=True)
		response.raise_for_status()
		with open(zip_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print(f"Downloaded to {zip_path}")

		# Unzip
		print(f"Unpacking {zip_path} ...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(models_dir)
		print(f"Unpacked to {models_dir}")

		os.remove(zip_path) # Optional: remove the zip file after extraction

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = ["llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(models_dir)==False or force_download==True:
			self.download_and_unpack(models_dir)
		
		llama.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
		llama.stats = self.stats
		print("loading model from", model_dir)
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
		self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		self.model.clean_layers_weights()
		self.model.eval()
		self.model.to(self.device)


def inference_chat():
	#sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 10
	sm, um, max_new_tokens = file_get_contents("./temp/85k_sample.txt"), "What's common between these article?", 20
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		#cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
		past_key_values = MyKVCache(len(model.model.layers), cache_folder="/media/mega4alik/ssd/kv_cache/") #HQQQuantizedCache
		print("\n\nGenerate started.", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", inputs.input_ids.shape)
		outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True).detach().cpu()
		answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)

#==============================================================================================

if __name__ == "__main__":
	device = torch.device("cuda:0")
	llama.loader = GDSWeights("/home/mega4alik/ssd/gds_export/manifest.json")
	stats = Stats()
	if 1==0: #prepare model without layers weights
		model_id = "meta-llama/Llama-3.2-1B-Instruct"
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		tokenizer.pad_token = tokenizer.eos_token
		model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #cpu | meta, dtype=should be bfloat16
		model.clean_layers_weights()
		model.save_pretrained("./models/llama3-1B") #saving model without layers weights
		tokenizer.save_pretrained("./models/llama3-1B"); exit()
	
	elif 2==2:
		model_id = "./models/llama3-1B-chat"
		print("loading model:", model_id)
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		model.clean_layers_weights()
		model.eval()
		model.cuda()
		inference_chat()








