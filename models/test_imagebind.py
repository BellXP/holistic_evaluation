import torch
from .llama_adapter import image_transform, load_model, format_prompt
from . import get_image


class TestImageBind:
    def __init__(self, model_name, device=None) -> None:
        ckpt_name = model_name.split('_')[1]
        model_path = f"/nvme/share/xupeng/llama_checkpoints/{ckpt_name}.pth"
        self.generator = load_model(model_path, device='cpu')
        self.img_transform = image_transform

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.device = device
        else:
            self.device = 'cpu'
        self.generator = self.generator.to(self.device)

    def generate(self, image, question, max_gen_len=256, temperature=0.1, top_p=0.75):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)

        prompts = [format_prompt(question)]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, input_type="vision")
        result = results[0].strip()

        return result
    
    def batch_generate(self, image_list, question_list, max_gen_len=256, temperature=0.1, top_p=0.75):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [format_prompt(question) for question in question_list]
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, input_type="vision")
        results = [result.strip() for result in results]

        return results