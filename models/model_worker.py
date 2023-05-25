import os
from PIL import Image
import cv2
import clip
import torch
import importlib


models_mae_path = '/home/xupeng/workplace/LLaMA-Adapter-v2/models_mae.py'
spec = importlib.util.spec_from_file_location('models_mae', models_mae_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mae_vit_base_patch16 = module.mae_vit_base_patch16
model_ckpt_path = '/home/xupeng/workplace/LLaMA-Adapter-v2/llama_adapter_v2_0518.pth'


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


class Model_Worker:
    def __init__(self, device=None) -> None:
        _, img_transform = clip.load("ViT-L/14")
        generator = mae_vit_base_patch16()
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        ckpt_model = ckpt['model']
        msg = generator.load_state_dict(ckpt_model, strict=False)

        self.img_transform = img_transform
        self.generator = generator

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            dtype = torch.float16
        elif type(device) is torch.device and 'cuda' in device.type:
            dtype = torch.float16
        else:
            device = 'cpu'
            dtype = torch.float32
        self.generator = self.generator.to(device, dtype=dtype)
        print(self.generator.device)
        print(type(self.generator.device))

    def generate(self, image, question, max_gen_len=64, temperature=0.1, top_p=0.75):
        if type(image) is str:
            img = cv2.imread(image)
            img = Image.fromarray(img)
        elif type(image) is Image.Image:
            pass
        else:
            raise NotImplementedError

        imgs = [img]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).cuda().half()

        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result
