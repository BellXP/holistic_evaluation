import torch

from .vpgtrans.common.config import Config
from .vpgtrans.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .vpgtrans.models import *
from .vpgtrans.processors import *

# from . import get_image
from PIL import Image

CFG_PATH = 'models/vpgtrans/vpgtrans_demo.yaml'


class TestVPGTrans:
    def __init__(self, device=None):
        cfg = Config(CFG_PATH)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.chat.prompt_templetae = (
            "###Human: <Img><ImageHere></Img> {question}"
            "###Assistant:"
        )

        if device is not None:
            self.move_to_device(device)
        self.model.eval()

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.chat.device = self.device
        self.model = self.model.to(self.device, dtype=self.dtype)
        # self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def generate(self, image, question):
        chat_state = CONV_VISION.copy()
        img_list = []
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            else:
                image = Image.fromarray(image)
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]

        return llm_message
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, *args, **kwargs):
        results = self.chat.batch_generate(image_list, question_list, *args, **kwargs)
        return results
