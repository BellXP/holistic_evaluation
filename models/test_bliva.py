import torch
from . import get_image
from .bliva.models import load_model_and_preprocess

# NOTE: BLIVA tend to generate other meaningless information after generating the required results, so we set its max_new_tokens into 256

# No sampling in default
class TestBLIVA:
    def __init__(self, device=None) -> None:
        device = 'cuda' if device is None else device
        model, vis_processors, _ = load_model_and_preprocess(name="bliva_vicuna", model_type="vicuna7b", is_eval=True, device='cuda')
        vis_processor = vis_processors["eval"]
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, do_sample=False, num_beams=1):
        imgs = [get_image(image)]
        imgs = [self.vis_processor(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [question]
        results = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens, use_nucleus_sampling=do_sample, num_beams=num_beams)
        result = results[0].strip()
        return result

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256, do_sample=False, num_beams=1):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processor(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = question_list
        with torch.cuda.amp.autocast():
            results = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens, use_nucleus_sampling=do_sample, num_beams=num_beams)
        results = [result.strip() for result in results]
        return results