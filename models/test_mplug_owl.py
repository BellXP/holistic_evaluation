import torch
from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from PIL import Image


class TestMplugOwl:
    def __init__(self, device=None):
        model_path='MAGAer13/mplug-owl-llama-7b'
        # model_path='checkpoints/huggingface_cache/hub/models--MAGAer13--mplug-owl-llama-7b/snapshots/057b523159d015b6c32c3b1f0340821d97930e1b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.prompt_template = (
            "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: <image>\n"
            "Human: {question}\n"
            "AI:"
        )

        if device is not None:
            self.move_to_device(device)
        self.model.eval()
        
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.device = device
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question):
        prompts = [f'''
            The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {question}
            AI: 
        ''']

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = Image.fromarray(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)


        return generated_text
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=128, *args, **kwargs):
        prompts = [self.prompt_template.format(question=x) for x in question_list]

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            # 'max_length': 128,
            'max_new_tokens': max_new_tokens,
        }

        if isinstance(image_list[0], str):
            images = [Image.open(x).convert('RGB') for x in image_list]
        else:
            images = [Image.fromarray(x) for x in image_list]
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return generated_text
