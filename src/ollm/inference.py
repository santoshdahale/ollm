import os, requests, zipfile
import torch
from transformers import AutoTokenizer
from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights
from . import llama
from . import gpt_oss

class Inference:
	def __init__(self, model_id, device="cuda:0"):
		self.model_id = model_id
		self.device = torch.device(device)
		self.stats = Stats()

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"llama3-1B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
			"llama3-3B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-3B-chat.zip",
			"llama3-8B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip",
			"gpt-oss-20B":    "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
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
		models_list = ["llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat", "gpt-oss-20B"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			self.download_and_unpack(models_dir)
		
		print("loading model from", model_dir)
		if self.model_id=="gpt-oss-20B":
			gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			gpt_oss.stats = self.stats
			self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		else:
			llama.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			llama.stats = self.stats			
			self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.model.clean_layers_weights()

		self.model.eval()
		self.model.to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

	
	def offload_layers_to_cpu(self, **args):
		self.model.offload_layers_to_cpu(**args)


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








