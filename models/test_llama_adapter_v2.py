import os
import importlib
from PIL import Image
from gradio_client import Client

import clip
import torch

from . import get_image, llama

from .model_mae import mae_vit_base_patch16

model_ckpt_path = 'checkpoints/llama_checkpoints/llama_adapter_v2_LORA-BIAS-7B.pth'


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class TestLLamaAdapterV2_web:
    def __init__(self, device=None) -> None:
        self.model = Client("http://106.14.127.192:8088/")
        self.max_length = 64
        self.temperature = 0.1
        self.top_p = 0.75

        if device is not None:
            self.move_to_device(device)
    
    def move_to_device(self, device):
        pass

    def generate(self, image, question: str):
        image = get_image(image)
        image_name = '.llama_adapter_v2_inference.png'
        image.save(image_name)
        output = self.model.predict(image_name, question, self.max_length, self.temperature, self.top_p, fn_index=1)
        
        return output


class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        # _, img_transform = clip.load("ViT-L/14", device=device)
        # generator = mae_vit_base_patch16()
        # ckpt = torch.load(model_ckpt_path, map_location='cpu')
        # ckpt_model = ckpt['model']
        # msg = generator.load_state_dict(ckpt_model, strict=False)

        llama_dir = 'checkpoints/llama_checkpoints'
        max_batch_size = int(os.environ.get('max_batch_size', 16))
        max_seq_len = int(os.environ.get('max_seq_len', 256))
        model, preprocess = llama.load(
            "LORA-BIAS-7B", llama_dir, device, download_root=llama_dir,
            max_seq_len=max_seq_len, max_batch_size=max_batch_size
        )
        model.eval()

        self.img_transform = preprocess
        self.generator = model

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.dtype = torch.float16
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.generator = self.generator.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_gen_len=256, temperature=0.1, top_p=0.75):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)

        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=128, temperature=0.1, top_p=0.75, *args, **kwargs):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = question_list
        prompt_template = kwargs.get('prompt_template', 'prompt_no_input')
        if prompt_template is not None:
            prompts = [PROMPT_DICT[prompt_template].format(instruction=x) for x in prompts]
        results = self.generator.generate(imgs, prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        results = [result.strip() for result in results]

        return results

    