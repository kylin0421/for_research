import jailbreakbench as jbb



#available parameters:
#method:DSN GCG JBC PAIR prompt_with_random_search
#model_name:"llama-2-7b-chat-hf" "vicuna-13b-v1.5" "gpt-3.5-turbo-1106"  "gpt-4-0125-preview"
#!!! Not every methods support all models listed above!
def load_jailbreakbench(method:str,model_name:str):

    artifact = jbb.read_artifact(
        method=method,
        model_name=model_name
    )
    #print(type(artifact.jailbreaks[75].jailbroken)) # The 75th index as an example

    prompts=[]

    for i in range(100):
        if artifact.jailbreaks[i].jailbroken:
            prompts.append(artifact.jailbreaks[i].prompt)


    return prompts