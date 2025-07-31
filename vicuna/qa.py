import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 只使用第0号GPU
import sys
sys.path.insert(0, "/home/u2021201665/code/baseline/llava-attack")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle,default_conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
  

def qa(args):
    #=====================准备数据====================
    device = args.device
    datamode = args.dataset
    model_path = os.path.expanduser("/home/u2021201665/share/llava-v1.6-vicuna-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base, model_name,device=device)
    model.to(device)



    # if datamode == "GQA":
    #     #选取的部分GQA问答问题
    #     with open("data/GQA.json", 'r', encoding='utf-8') as file:
    #         # 解析 JSON 数据
    #         data = json.load(file)
    # elif datamode == "SafetyBench0":
    #     #是MM-safetybench的问题加图片，筛选了主题
    #     with open("data/SafetyBench0.json",'r',encoding='utf-8') as file:
    #         data = json.load(file)
    # elif datamode == "SafetyBench3":
    #     with open("data/SafetyBench3.json",'r',encoding='utf-8') as file:
    #         data = json.load(file)
    
    # elif datamode == "pixart" or datamode == "pixart-des":
    #     #用pixart生成的图片，问题手写得到
    #     with open(f"data/{datamode}.json",'r',encoding='utf-8') as file:
    #         data = json.load(file)

    # elif datamode == "deepseekvoid" or datamode == "deepseekvoid-none":
    #     #deepseek是由deepseek得到的问题，void代表空白图片，None代指无图片
    #     with open(f"data/{datamode}.json",'r',encoding='utf-8') as file:
    #         data = json.load(file)

    # elif datamode == "SafetyBenchvoid" or datamode == "SafetyBenchvoid-none":
    #     #None是测试无图片的影响，void是图片都是空白的背景
    #     with open(f"data/{datamode}.json",'r',encoding='utf-8') as file:
    #         data = json.load(file)
    #     # print(len(data))
    #     # exit(0)
    #     qa = [data[str(i)] for i in range(len(data))]
    #     questions = [i['question'] for i in qa]
    #     images = [i['image'] for i in qa]
    # elif datamode =="GQA-des" or datamode =="SafetyBench0-des":
    #     #des是对图片描述的任务，用于记录图片对分类器的影响
    #     with open(f"data/{datamode}.json",'r') as f:
    #         data = json.load(f)
    
    print(datamode)
    if "void" in datamode:
        temp_datamode = datamode.split('-void')[0]
        with open(f"instructions/{temp_datamode}.json",'r',encoding='utf-8') as file:
            data = json.load(file)
        for i in list(data.keys()):
            data[i]['image'] = None
    else:
        with open(f"instructions/{datamode}.json",'r',encoding='utf-8') as file:
            data = json.load(file)

    #记录每层的中间状态与回答 
    hidden_states = []
    responses = []
    output_hidden_states = args.output_hidden_states
    num = 0
    for i in tqdm(list(data.keys()),desc="quesntion-answering",unit="image"):
        print(num)
        num += 1
        #======================
        qs = data[i]['question']
        
        image_file = data[i]['image']
        
        if image_file is not None:
            image_file = image_file.replace("/home/yifan/liangshuang/asset","/home/u2021201665/asset")
            image_file = image_file.replace("/high_perf_store/surround-view/liangshuang/LLaVA-CAV","/home/u2021201665/asset")
            image_file = image_file.replace("/home/yifan/liangshuang/code/baseline/","/home/u2021201665/code/baseline/")
            
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
 
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        # a = torch.load("/home/u2021201665/asset/QA-vicuna/gpt_answer.pth")
        # print(a.shape)
        # print(a[0,:,:])
        # print(image_tensor.shape)
        # exit(0)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        data[i]['response'] = outputs

    # 保存为 JSON 文件
    with open(f'instructions/{datamode}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/u2021201665/weights/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--output_hidden_states",type=bool,default=True)
    parser.add_argument("--device",type=str,default='cuda')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    qa(args)



