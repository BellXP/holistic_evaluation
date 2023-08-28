import sys
from PIL import Image

import torch

from omegaconf import OmegaConf
sys.path.append('models/Cheetah')
from cheetah.common.config import Config
from cheetah.common.registry import registry
from cheetah.conversation.conversation import Chat
from cheetah.models import *
from cheetah.processors import *


class TestCheetah:

    def __init__(
        self,
        cfg_path: str='models/Cheetah/eval_configs/cheetah_eval_vicuna.yaml',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        config = OmegaConf.load(cfg_path)
        cfg = Config.build_model_config(config)
        model_cls = registry.get_model_class(cfg.model.arch)
        model = model_cls.from_config(cfg.model).to(device)

        vis_processor_cfg = cfg.preprocess.vis_processor.eval
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device=device)
        self.chat = chat

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=16, *args, **kwargs):
        if type(image_list[0]) is not str:
            images = [Image.fromarray(x) for x in image_list]
        else:
            images = [Image.open(img).convert('RGB') for img in image_list]
        images = [[x] for x in images]
        results = self.chat.batch_answer(
            images, question_list, max_new_tokens=max_new_tokens
        )
        return results
