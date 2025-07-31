from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

class Model:
    def __init__(self,pretrained,device='cuda',dtype=torch.float16):
        self.device = device
        self.processor = LlavaNextProcessor.from_pretrained(pretrained)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(pretrained, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
        self.norm = self.model.language_model.model.norm
        self.layers = 32
        self.model.eval()
        self.original_outputs = []
        self.perturbed_outputs = []
        self.hooks = []
    
    def generate(self,query,image,output_hidden_states=True):
        # prepare image and text prompt, using the appropriate prompt template
        if image != None:
            image = Image.open(image)
            conversation = [ 
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                    ],
                },
            ] 
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # print(prompt)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        else:
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # print(prompt)
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        # print(inputs)
        with torch.no_grad():
            # autoregressively complete prompt
            output = self.model.generate(**inputs, max_new_tokens=128)
            if output_hidden_states:
                states = self.model(**inputs,output_hidden_states=output_hidden_states)


        input_length = inputs["input_ids"].shape[1]  # 输入的长度
        generated_ids = output[:, input_length:]  # 只保留生成的部分
        # 解码生成的部分
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(answer)
        # exit(0)
        if output_hidden_states:
            return answer,states.hidden_states
        else:
            return answer
    def forward(self,query,image,output_hidden_states=True):
        # prepare image and text prompt, using the appropriate prompt template
        if image is not None:
            image = Image.open(image)
            # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
            # Each value in "content" has to be a list of dicts with types ("text", "image") 
            conversation = [ 
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                    ],
                },
            ] 
            
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # print(prompt)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        else:
            # print(0)
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # print(prompt)
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():          
            output = self.model(**inputs, output_hidden_states=output_hidden_states) 
        return output.hidden_states

