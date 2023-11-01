import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf
from .cheetah.common.config import Config
from .cheetah.common.registry import registry
from .cheetah.conversation.conversation import Chat, CONV_VISION

from .cheetah.models import *
from .cheetah.processors import *

from . import DATA_DIR, get_image


class TestCheetah:
    def __init__(self, device=None) -> None:
        cfg_path = 'models/cheetah/cheetah_eval_vicuna.yaml'
        config = OmegaConf.load(cfg_path)
        config['model']['ckpt'] = f"{DATA_DIR}/{config['model']['ckpt']}"
        cfg = Config.build_model_config(config)
        cfg['model']['llama_model'] = f"{DATA_DIR}/{cfg['model']['llama_model']}"
        model_cls = registry.get_model_class(cfg.model.arch)
        model = model_cls.from_config(cfg.model).to('cuda')

        vis_processor_cfg = cfg.preprocess.vis_processor.eval
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda')

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=1024, do_sample=False, num_beams=1):
        prompt = f"<Img><HereForImage></Img> {question}"
        output = self.chat.answer([image], prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1024, do_sample=False, num_beams=1):
        images = [[get_image(image)] for image in image_list]
        prompts = [f"<Img><HereForImage></Img> {question}" for question in question_list]
        output = self.chat.batch_answer(images, prompts, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        return output