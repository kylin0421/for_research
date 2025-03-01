import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



#This program simply provide a function to utilize, do NOT run this program directly!!!

def get_activation(model, layer_name, prompt, tokenizer,device,return_text:bool=False):
    """
    获取指定层的激活值
    
    参数:
        model: LlamaForCausalLM 模型实例
        layer_name: 字符串，例如 "model.layers.5.mlp"
        prompt: 输入文本
        tokenizer: 分词器
    
    返回:
        activation: 该层的激活值张量
    """
    activations = {}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 获取 prompt 的 token 长度
    input_length = inputs["input_ids"].shape[-1]

    def hook_fn(module, input, output):
        activations['value'] = output
    
    # 根据layer_name找到对应模块
    
    hook = layer_name.register_forward_hook(hook_fn)
    
    
    # 模型推理
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        do_sample=False,      
        pad_token_id=tokenizer.eos_token_id,
        max_length=16384
    )
    
    
    # 移除hook
    hook.remove()


    if return_text:
        return activations['value'][0],tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    else:
        return activations['value'][0]

# 使用示例:
# activation = get_activation(
#     model, 
#     "model.layers.5.mlp",  # 或 "model.layers.5.self_attn" 等
#     "Your input text", 
#     tokenizer
# )