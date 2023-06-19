import torch
from transformers import CLIPImageProcessor
from .instruct_blip.models import load_model_and_preprocess
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from . import get_image
from PIL import Image

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
    'prompt_lavis': "Question: {instruction} Short answer:",
}

class TestInstructBLIP:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device='cpu')
        if device is not None:
            self.move_to_device(device)
        self.model.eval()

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
            convert_weights_to_fp16(self.model.visual_encoder)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.llm_model = self.model.llm_model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question})[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, *args, **kwargs):
        if type(image_list[0]) is not str:
            imgs = [Image.fromarray(x) for x in image_list]
        else:
            imgs = [Image.open(img).convert('RGB') for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = question_list
        prompt_template = kwargs.get('prompt_template', None)
        if prompt_template is not None:
            prompts = [PROMPT_DICT[prompt_template].format(instruction=x) for x in prompts]
        results = self.model.generate({"image": imgs, "prompt": prompts})
        results = [result.strip() for result in results]

        return results
