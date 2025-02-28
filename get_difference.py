import json
import torch

method="GCG"


file1_name=f"{method}/act_success_{method}.json"
file2_name=f"{method}/act_nosuffix_{method}.json"

with open(file1_name, "r", encoding="utf-8") as file:
    list_of_lists_1=json.load(file)

with open(file2_name, "r", encoding="utf-8") as file:
    list_of_lists_2=json.load(file)


tensor1 = torch.tensor(list_of_lists_1, dtype=torch.float16)
tensor2 = torch.tensor(list_of_lists_2, dtype=torch.float16)

assert tensor1.shape==tensor2.shape, "The twotensors are not in the same shape, please check if the two json files are from the same method and model!"

r_tensor=tensor1-tensor2

r_nested_list=r_tensor.tolist()


difference_store=f"{method}/difference_success_nosuffix.json"
with open(difference_store, "w", encoding="utf-8") as file:
    json.dump(r_nested_list, file, indent=4)