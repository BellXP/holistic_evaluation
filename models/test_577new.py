import torch
from . import get_image, get_BGR_image, DATA_DIR
from . import llama_577new as llama

llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_path = f'{DATA_DIR}/llama_checkpoints/577new/BIAS_LORA_NORM-336-Chinese-74-7B.pth'  # 'BIAS_LORA_NORM-336-Chinese-626-7B.pth'


class Test577new:
    def __init__(self, device=None) -> None:
        self.model, self.img_transform = llama.load(model_path, llama_dir, device='cuda', max_seq_len=256, max_batch_size=16)
        self.model.eval()

        if device is not None:
            self.device = 'cuda'
            # self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.device = device
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question)]
        results = self.model.generate(imgs, prompts)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]

        return results