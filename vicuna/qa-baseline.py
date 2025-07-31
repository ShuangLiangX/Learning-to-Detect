import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image

import sys 
sys.path.insert(0,"/home/u2021201665/code/baseline/llava-attack")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle,default_conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import time

def qa(args):
    #=====================准备数据====================
    datamode = args.dataset
    device = args.device
    
    model_path = os.path.expanduser("/home/u2021201665/share/llava-v1.6-vicuna-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,device=device)
    model.to(device)


    
    with open(f"../HiddenData/{datamode}.json",'r',encoding='utf-8') as file:
        data = json.load(file)
    #记录每层的中间状态与回答 
    
    hidden_states = []
    with open(f"/home/u2021201665/code/baseline/llava-attack/adv_suffix.json",'r') as f:
        suffixes = json.load(f) 
    last_line = suffixes[-1]

    image_tensor = None
    start = time.perf_counter()
    for i in tqdm(list(data.keys())[:218],desc="quesntion-answering",unit="image"):
        #======================
        # print(i)
        qs = data[i]['question']
        
        image = data[i]['image']

        if "vajm" in datamode:
            image = "/home/u2021201665/code/baseline/llava-attack/vajm-vicuna/bad_prompt.bmp"

        elif "umk" in datamode:
            qs = qs + ' | '+last_line
            image = "/home/u2021201665/code/baseline/llava-attack/umk-vicuna/bad_prompt.bmp"
                
        if image is not None:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:  
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        

        if '.bmp' in image:
            image = Image.open(image).convert('RGB')
            image_patches = [image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]]
            image_tensor = torch.stack(image_patches,dim=0)
        else:
            image = Image.open(image).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        input_ids = input_ids.to(device)
        attention_mask =  torch.LongTensor( [ [1]* (input_ids.shape[1])]).to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                images=image_tensor.half(),
                image_sizes=[image.size],
                output_hidden_states=True
            )

        extracted_states = [tensor[:, -1, :] for tensor in outputs.hidden_states]
        # 将提取的向量堆叠成一个新的张量（形状为 [N, 4096]，N 是 tuple 中张量的数量）
        extracted_tensor = torch.cat(extracted_states, dim=0)

        hidden_states.append(extracted_tensor)

    hidden_states = torch.stack(hidden_states) 
    end = time.perf_counter()
    print(f"运行时间: {end - start:.6f} 秒")
    exit(0)
    print(hidden_states.shape)

    torch.save(hidden_states,f"/home/u2021201665/asset/QA-vicuna/{datamode}_answer.pth")      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--device",type=str,default='cuda:7')
    args = parser.parse_args()
    qa(args)
