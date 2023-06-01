import torch
from . import get_image
from . import llama_adapter_v3 as llama

llama_dir = '/nvme/share/xupeng/llama_checkpoints'
model_path = '/nvme/share/xupeng/llama_checkpoints/llama-adapter-v3-400M-30token-pretrain100-finetune3/converted_checkpoint-0.pth'


class TestLLamaAdapterV3:
    def __init__(self, device=None) -> None:
        self.model, self.img_transform = llama.load(model_path, llama_dir, device='cpu')

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.device = device
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)

    def generate(self, image, question):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question)]
        results = self.model.generate(imgs, prompts)
        result = results[0].strip()

        return result
    
    def batch_generate(self, image_list, question_list):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts)
        results = [result.strip() for result in results]

        return results