import torch
from transformers import CLIPImageProcessor
from .instruct_blip.models import load_model_and_preprocess
from .instruct_blip.models.eva_vit import convert_weights_to_fp16


class TestInstructBLIP:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device='cpu')

        if device is not None:
            self.move_to_device(device)

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

    def generate(self, image, question):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question})[0]

        return output
    