import torch
import json

def test_data(device="cuda"):
    
    with open("/code/vicuna/instructions/GQA-test.json",'r') as f:
        data_safe = json.load(f)
    with open("/code/vicuna/instructions/advbench-test.json",'r') as f:
        data_unsafe = json.load(f)        
    #
    safe = torch.load("/code/asset/HiddenStates/GQA_answer.pth", map_location=device)
    unsafe = torch.load("/code/asset/HiddenStates/advbench_answer.pth", map_location=device)

    #   
    safe_indice = [int(i) for i in list(data_safe.keys())]
    unsafe_indice = [int(i)-246 for i in list(data_unsafe.keys())]
    safety = safe[safe_indice]
    unsafety = unsafe[unsafe_indice]
    return safety, unsafety

def train_data(device="cuda"):
    with open("/code/vicuna/instructions/GQA-train.json",'r') as f:
        data_safe = json.load(f)
    with open("/code/vicuna/instructions/advbench-train.json",'r') as f:
        data_unsafe = json.load(f)        

    safe = torch.load("/code/asset/HiddenStates/GQA_answer.pth", map_location=device)
    unsafe = torch.load("/code/asset/HiddenStates/advbench_answer.pth", map_location=device)
    print("train data")
        
    safe_indice = [int(i) for i in list(data_safe.keys())]
    unsafe_indice = [int(i)-246 for i in list(data_unsafe.keys())]
    safety = safe[safe_indice]
    unsafety = unsafe[unsafe_indice]
    return safety,unsafety

