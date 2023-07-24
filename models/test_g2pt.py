import torch
from . import get_image, DATA_DIR, g2pt

llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_paths = {
    'G2PT-7B': "/mnt/data/pjlab-3090-gvadapt/vlm_eval/minimal-gpt-finetune/falcon_pretrain/output/finetune/qformerv2peft_bs4_acc1_epoch4_lr5e-5_mlr5e-6-wd0.02-pre190000/epoch3/consolidated.00-of-01.model.pth",
    'G2PT-3B': "/mnt/data/pjlab-3090-gvadapt/vlm_eval/minimal-gpt-finetune/falcon_pretrain/output/finetune/qformerv2peft_3B_bs4_acc1_epoch4_lr5e-5_mlr5e-6-wd0.02/epoch3/consolidated.00-of-01.model.pth",
    'G2PT-13B': "/nvme/share/VLP_web_data/G2PT-13B"
}


class TestG2PT:
    def __init__(self, model_name, device) -> None:
        model_path = model_paths[model_name]
        llama_type = model_name.split('-')[-1]
        self.model, self.img_transform = g2pt.load(model_path, llama_type, llama_dir, device, max_seq_len=256, max_batch_size=8)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, temperature=0, top_p=0.75):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [g2pt.format_prompt(question)]
        with torch.cuda.amp.autocast():
            results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256, temperature=0, top_p=0.75):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [g2pt.format_prompt(question) for question in question_list]
        with torch.cuda.amp.autocast():
            results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        results = [result.strip() for result in results]

        return results

    
