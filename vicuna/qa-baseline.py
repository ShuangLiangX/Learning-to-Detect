import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle,default_conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def qa(args):


    device = args.device    
    model_path = str(REPO_ROOT / "asset/weights/llava-v1.6-vicuna-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,device=device)
    model.to(device)
    for datamode in args.datasets:
        print(datamode)
        metadata_path = REPO_ROOT / "Benchmarks" / f"{datamode}.json"
        if not metadata_path.exists():
            metadata_path = REPO_ROOT / "vicuna/instructions" / f"{datamode}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Provide Benchmarks/{datamode}.json with question and image fields "
                f"before extracting {datamode} hidden states."
            )
        with metadata_path.open(encoding="utf-8") as file:
            data = json.load(file)

        hidden_states = []
        with open(REPO_ROOT / "Benchmarks/umk_suffix-vicuna.json", "r") as f:
            suffixes = json.load(f) 
        last_line = suffixes[-1]
        image_tensor = None
        for i in tqdm(list(data.keys()),desc="quesntion-answering",unit="image"):
            qs = data[i]['question']
            image = data[i]['image']

            if datamode == "SafetyBench-vajm":
                image = "asset/adversarial_images/vajm-vicuna.bmp"
            elif datamode == "SafetyBench-umk":
                image = "asset/adversarial_images/umk-vicuna.bmp"
                qs = qs + ' | '+last_line

            if image is not None and not os.path.isabs(image):
                image = str(REPO_ROOT / image)
                    
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
                image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
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


        torch.save(hidden_states, REPO_ROOT / "asset/HiddenStates" / f"{datamode}_answer.pth")
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["FC", "JOOD", "HADES", "MML-m", "SafetyBench-vajm", "SafetyBench-umk", "mm-vet"],
        help="Benchmark names to extract; mm-vet requires separately supplied metadata and images.",
    )
    args = parser.parse_args()
    set_seed()
    qa(args)
