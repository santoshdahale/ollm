# qwen3_next_export
import json, os
import torch
from safetensors.torch import safe_open, save_file
from ollm import file_get_contents

def generate_manifest():
    d = {}
    for layer_idx in range(num_hidden_layers):
        base = f"model.layers.{layer_idx}."
        d[base] = []
        for manifest_name, filename in wmap.items():
            if manifest_name.startswith(base) and ".mlp.experts." not in manifest_name:
                attr_name = manifest_name.replace(base, "")
                d[base].append(attr_name)

    for layer_idx in range(num_hidden_layers):
        for expert_idx in range(num_experts):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
            d[base] = []
            for manifest_name, filename in wmap.items():
                if manifest_name.startswith(base):
                    attr_name = manifest_name.replace(base, "")
                    d[base].append(attr_name)

    with open(f"{out_dir}manifest.json", "w") as f: json.dump(d, f, indent=4)


#====================================
path, out_dir = "/media/mega4alik/ssd2/models/qwen3_next/", "/home/mega4alik/Desktop/models/qwen3_next/"
wmap = json.loads(file_get_contents(f"{path}model.safetensors.index.json"))["weight_map"]
num_hidden_layers, num_experts = 48, 512
exit() #DONE
#generate_manifest()

#export non layer weights like embed_tokens, lm_head
if 1==2:
    d = {}
    for manifest_name, filename in wmap.items():
        if not "model.layers" in manifest_name and "mtp" not in manifest_name:
            with safe_open(path+filename, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(manifest_name)
                d[manifest_name] = tensor
                print(manifest_name)
    save_file(d, out_dir+"model.safetensors")

# export layers
for layer_idx in range(num_hidden_layers):
    base, d = f"model.layers.{layer_idx}.", {}
    for manifest_name, filename in wmap.items():
        if manifest_name.startswith(base) and ".mlp.experts." not in manifest_name:
            attr_name = manifest_name.replace(base, "")
            with safe_open(path+filename, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(manifest_name)
                d[attr_name] = tensor
                print(layer_idx, attr_name)
    torch.save(d, out_dir+base.replace(".","__")+".pt")


# export mlp.experts
for layer_idx in range(num_hidden_layers):
    for expert_idx in range(num_experts):
        base, d = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.", {}
        for manifest_name, filename in wmap.items():
            if manifest_name.startswith(base):
                attr_name = manifest_name.replace(base, "")
                with safe_open(path+filename, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(manifest_name)
                    d[attr_name] = tensor
                    print(layer_idx, expert_idx, attr_name)
        torch.save(d, out_dir+base.replace(".","__")+".pt")
