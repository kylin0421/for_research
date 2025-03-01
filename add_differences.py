import torch
import json

method1="DSN"
method2="GCG"
method3="JBC"
method4="PAIR"
feature1="success"
feature2="nosuffix"


#file1_name=f"{method1}/difference_{feature1}_{feature2}.json"
#file2_name=f"{method2}/difference_{feature1}_{feature2}.json"

with open("DSN/difference_success_nosuffix.json", "r", encoding="utf-8") as file:
    list_of_lists_1=json.load(file)

with open("GCG/difference_success_nosuffix.json", "r", encoding="utf-8") as file:
    list_of_lists_2=json.load(file)


with open("JBC/difference_success_nosuffix.json", "r", encoding="utf-8") as file:
    list_of_lists_3=json.load(file)


with open("PAIR/difference_success_nosuffix.json", "r", encoding="utf-8") as file:
    list_of_lists_4=json.load(file)





tensor1 = torch.tensor(list_of_lists_1, dtype=torch.float16)
tensor2 = torch.tensor(list_of_lists_2, dtype=torch.float16)
tensor3 = torch.tensor(list_of_lists_3, dtype=torch.float16)
tensor4 = torch.tensor(list_of_lists_4, dtype=torch.float16)

#print(tensor1.shape)
#print(tensor2.shape)
#print(tensor3.shape)
#print(tensor4.shape)

#61,49,91,38


r_tensor=torch.cat([tensor1,tensor2,tensor3,tensor4],dim=0)


r_nested_list=r_tensor.tolist()

r_store="sum_of_4.json"
with open(r_store, "w", encoding="utf-8") as file:
   json.dump(r_nested_list, file, indent=4)



    