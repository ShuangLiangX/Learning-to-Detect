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

def qa(args):
    #=====================准备数据====================
    datamode = args.dataset
    device = args.device
    
    model_path = os.path.expanduser("/home/u2021201665/share/llava-v1.6-vicuna-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device=device)
    model.to(device)


    
    with open(f"{datamode}",'r',encoding='utf-8') as file:
        data = json.load(file)
    # with open("/home/u2021201665/code/HiddenData/SafetyBench-vajm",'r') as f:
    #     data = json.load(f)
    #记录每层的中间状态与回答 
    # types = "image-level"
    # types="vajm"
    types = args.types
    num = 0
    for i in tqdm(list(data.keys()),desc="quesntion-answering",unit="image"):
        #======================
        # print(i)
        print(num)
        num += 1
        qs = data[i]['question']
        if types == "image-level":
            image = "/home/u2021201665/code/SCAV-vicuna/vajm-1.0/bad_prompt.bmp"
        elif types == "vajm":
            image = "/home/u2021201665/code/baseline/llava-attack/vajm-vicuna/bad_prompt.bmp"
            qs = 'Please extend the following sentence: '+qs
        else:
            print("wrong type")
            exit(0)
        if image is not None:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:  
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
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

        # print(image_tensor.shape)
        # exit(0)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids.to(device),
                images=image_tensor.unsqueeze(0).to(device).half(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True)
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print('>>>', response)
        data[i][types] = response
    # with open(f"{datamode}",'w',encoding='utf-8') as file:
    #     json.dump(data,file,indent=4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--device",type=str,default='cuda:7')
    parser.add_argument("--types",type=str,required=True)
    
    
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()
    qa(args)
