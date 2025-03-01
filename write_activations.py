import json
from get_activation import get_activation
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

model_name = "lmsys/vicuna-7b-v1.5-16k"
device="cuda:5"

start_time = time.time()

# 加载 tokenizer 和模型（FP16）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
              
).to(device)

end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds")



method="PAIR"
feature="safe"
with open(f"{method}/{feature}_{method}_vicuna7b.json", "r", encoding="utf-8") as file:
    prompts=json.load(file)

print(f"{len(prompts)} prompts loaded, start getting activations...")

model=model
layer_name=model.model.layers[16]
tokenizer=tokenizer

activations=[]
count=0

start_time=time.time()
for prompt in prompts:
    time_1=time.time()
    activation=get_activation(model,layer_name,prompt,tokenizer,device=device,return_text=False)
    activations.append(activation.tolist())
    time_2=time.time()
    print(f"{count}th activation loaded, using {time_2-time_1:.2f} seconds")
    count+=1
end_time=time.time()
print(f"All {len(activations)} activations loaded in {end_time - start_time:.2f} seconds")


start_time=time.time()
with open(f"{method}/act_{feature}_{method}.json", "w", encoding="utf-8") as file:
            json.dump(activations, file, indent=4)
end_time=time.time()
print(f"writing data into json using {end_time - start_time:.2f} seconds")