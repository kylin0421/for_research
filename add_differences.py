import torch
import json

method1="DSN"
method2="GCG"
feature1="success"
feature2="nosuffix"


#file1_name=f"{method1}/difference_{feature1}_{feature2}.json"
#file2_name=f"{method2}/difference_{feature1}_{feature2}.json"

with open("DSN/act_safe_DSN.json", "r", encoding="utf-8") as file:
    list_of_lists_11=json.load(file)

with open("DSN/act_nosuffix_DSN.json", "r", encoding="utf-8") as file:
    list_of_lists_12=json.load(file)


with open("DSN/act_success_DSN.json", "r", encoding="utf-8") as file:
    list_of_lists_13=json.load(file)


with open("GCG/act_safe_GCG.json", "r", encoding="utf-8") as file:
    list_of_lists_21=json.load(file)

with open("GCG/act_nosuffix_GCG.json", "r", encoding="utf-8") as file:
    list_of_lists_22=json.load(file)


with open("GCG/act_success_GCG.json", "r", encoding="utf-8") as file:
    list_of_lists_23=json.load(file)



tensor11 = torch.tensor(list_of_lists_11, dtype=torch.float16)
tensor12 = torch.tensor(list_of_lists_12, dtype=torch.float16)
tensor13 = torch.tensor(list_of_lists_13, dtype=torch.float16)
tensor21 = torch.tensor(list_of_lists_21, dtype=torch.float16)
tensor22 = torch.tensor(list_of_lists_22, dtype=torch.float16)
tensor23 = torch.tensor(list_of_lists_23, dtype=torch.float16)



r_tensor=torch.cat([tensor11,tensor12,tensor13,tensor21,tensor22,tensor23],dim=0)


r_nested_list=r_tensor.tolist()

r_store="sum_of_6.json"
with open(r_store, "w", encoding="utf-8") as file:
    json.dump(r_nested_list, file, indent=4)



    