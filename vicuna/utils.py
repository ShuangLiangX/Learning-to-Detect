#===============hidden states的提取================
import torch
import json

def test_data(device="cuda",void=False):
    
    with open("/home/u2021201665/code/SCAV-vicuna/instructions/GQA-test.json",'r') as f:
        data_safe = json.load(f)
    with open("/home/u2021201665/code/SCAV-vicuna/instructions/advbench-test.json",'r') as f:
        data_unsafe = json.load(f)
        

    safe = torch.load("GQA_answer.pth", map_location=device)
    unsafe = torch.load("advbench_answer.pth", map_location=device)


    safe_indice = [int(i) for i in list(data_safe.keys())]
    # print(safe_indice)
    unsafe_indice = [int(i) for i in list(data_unsafe.keys())]
    # 从 safe_indice 中随机挑选 10 个作为训练集
    # print(unsafe_indice)
    safety = safe[safe_indice]
    unsafety = unsafe[unsafe_indice]
    return safety, unsafety
def train_data(device="cuda",void=False):
    # print(device)
    # exit(0)
    # print(1)
    with open("/home/u2021201665/code/SCAV-vicuna/instructions/GQA-train.json",'r') as f:
        data_safe = json.load(f)
    with open("/home/u2021201665/code/SCAV-vicuna/instructions/advbench-train.json",'r') as f:
        data_unsafe = json.load(f)
        
    if void:
        safe = torch.load("/home/u2021201665/asset/QA-vicuna/GQA-void_answer.pth", map_location=device)
        unsafe = torch.load("/home/u2021201665/asset/QA-vicuna/advbench-void_answer.pth", map_location=device)
    else:
        safe = torch.load("/home/u2021201665/asset/QA-vicuna/GQA_answer.pth", map_location=device)
        unsafe = torch.load("/home/u2021201665/asset/QA-vicuna/advbench_answer.pth", map_location=device)
    print("train data")
    #提取训练tensor
    
    safe_indice = [int(i) for i in list(data_safe.keys())]
    unsafe_indice = [int(i)-246 for i in list(data_unsafe.keys())]
    safety = safe[safe_indice]
    unsafety = unsafe[unsafe_indice]
    return safety,unsafety


def normalize(images,device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images,device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


