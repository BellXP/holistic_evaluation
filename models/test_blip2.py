import torch
import contextlib
from lavis.models import load_model_and_preprocess
from . import get_image


def maybe_autocast(dtype=None):
    return contextlib.nullcontext()


class TestBlip2:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )

        if not torch.cuda.is_bf16_supported():
            self.model.maybe_autocast = maybe_autocast

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float32
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    def generate(self, image, question):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        })

        return answer[0]
    