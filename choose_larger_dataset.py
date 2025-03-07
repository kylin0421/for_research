#venv name vicuna-torch

import json
import pickle
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from get_activation import get_activation
from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge



api_key = ""
jailbreak_judge = Llama3JailbreakJudge(api_key)
refusal_judge = Llama3RefusalJudge(api_key)
device=torch.device("cuda:5")

# 检查 CUDA
assert torch.cuda.is_available(), "CUDA not available!"
#print(f"Using device: cuda")
#print(f"Torch version: {torch.__version__}")

# 模型路径
model_name = "lmsys/vicuna-7b-v1.5-16k"

start_time = time.time()

# 加载 tokenizer 和模型（FP16）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 FP16
    
).to(device)

end_time = time.time()

#print("model:",model)
#print("named_modules:",model.named_modules())


print(f"Model loaded in {end_time - start_time:.2f} seconds")

# 验证设备
#print(f"Model device: {next(model.parameters()).device}")



# attention的输出
#f"model.layers.{n}.self_attn.o_proj"

# mlp的输出
#f"model.layers.{n}.mlp.down_proj"

# 整层的输出，直接用层名即可
#f"model.layers.{n}"

layer_name=model.model.layers[16]



method="JBC"
used_model_name="vicuna-13b-v1.5"

prompts=load_jailbreakbench(method,used_model_name)

num_prompt=0   #starting from the smallest index unprocessed prompt in the last run!
success=0
success_prompts=set()
for prompt in prompts[num_prompt:]:

    prompt=prompt+"\nAnswer:"
    
    for num_try in range(3):
        
         # 输入移到 GPU

        start_time = time.time()

        activation,output=get_activation(model,layer_name,prompt,tokenizer,return_text=True)
        #print(f"{num_prompt}th activation:",activation)
        #print(f"{num_prompt}th shape:",activation.shape)
        #print(f"{num_prompt}th generated:",output)
        end_time = time.time()

        print(f"Get {num_prompt}th prompt {num_try+1}th try activation and output time: {end_time - start_time:.2f} seconds")
        is_jailbroken = jailbreak_judge([prompt], [output])
        is_refusal = refusal_judge([prompt], [output])
        print(f"jailbreak:{is_jailbroken}, refusal:{is_refusal}")
        if is_jailbroken[0] or not is_refusal[0]:
            print("suspect response:",output)

        len_before=len(success_prompts)


        if is_jailbroken[0]:
            success+=1
            success_prompts.add(prompt)
            break

    

    start_time = time.time()
    if len_before!=len(success_prompts):
        with open(f"{method}/success_{method}_vicuna7b.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
        existing_data.extend(success_prompts)
        existing_data=list(set(existing_data))
        with open(f"{method}/success_{method}_vicuna7b.json", "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)
    end_time = time.time()

    num_prompt+=1

print(f"All prompts generated by {method} for {used_model_name} successfully jailbreaking on {model_name} saved!!!")



'''

# 推理
start_time = time.time()
outputs = model(**inputs)
end_time = time.time()


print("special token map:",tokenizer.special_tokens_map)




print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"Generated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


'''