import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


model_name = "lmsys/vicuna-7b-v1.5-16k"



# load tokenizer and model(FP16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    
)



print(f"{model.config.model_type}_max_position_embeddings:",model.config.max_position_embeddings)