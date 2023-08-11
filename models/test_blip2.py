import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from lavis.models.eva_vit import convert_weights_to_fp16
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

def maybe_autocast(dtype=None):
    return contextlib.nullcontext()

def new_maybe_autocast(self, dtype=None):
    if torch.cuda.is_bf16_supported() and dtype is torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()

class TestBlip2:
    def __init__(self, device=None) -> None:
        self.device = device
        self.dtype = torch.float32
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )

        # if not torch.cuda.is_bf16_supported():
        #     self.model.maybe_autocast = maybe_autocast
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

        # if device is not None:
        #     self.move_to_device(device)
        self.model.eval()
        self.model.to(self.device)

    # def move_to_device(self, device):
    #     if device is not None and 'cuda' in device.type:
    #         self.dtype = torch.float16
    #         self.device = device
    #         convert_weights_to_fp16(self.model.visual_encoder)
    #     else:
    #         self.dtype = torch.float32
    #         self.device = 'cpu'
    #         self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
    #     self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question):
        image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        })

        return answer[0]

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, *args, **kwargs):
        if type(image_list[0]) is not str:
            imgs = [Image.fromarray(x) for x in image_list]
        else:
            imgs = [Image.open(img).convert('RGB') for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompt_template = kwargs.get('prompt_template', None)
        prompts = question_list
        if prompt_template is not None:
            prompts = [PROMPT_DICT[prompt_template].format(instruction=x) for x in prompts]
        max_new_tokens = kwargs.get('max_new_tokens', 16)
        results = self.model.generate(
            {"image": imgs, "prompt": prompts,},
            max_length=max_new_tokens
        )
        results = [result.strip() for result in results]

        return results
    