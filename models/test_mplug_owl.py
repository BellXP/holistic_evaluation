import torch
from transformers import AutoTokenizer
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from . import get_image

prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI: "


class TestMplugOwl:
    def __init__(self, device):
        self.device = device
        pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
        # pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b-ft'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        )
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=1024, do_sample=False, num_beams=1):
        prompts = [prompt_template.format(question)]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': do_sample,
            'top_k': 5,
            'max_length': max_new_tokens,
            'num_beams': num_beams
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=1024):
        prompts = [question]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens,
            'num_beams': 1
        }

        import time
        begin = time.time()
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        gen_cost = time.time() - begin
        print(f"Time cost of generation: {gen_cost}")
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256, do_sample=False, num_beams=1):
        images = [get_image(image) for image in image_list]
        prompts = [prompt_template.format(question) for question in question_list]
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': do_sample,
            'top_k': 5,
            'max_length': max_new_tokens,
            'num_beams': num_beams
        }

        import time
        begin = time.time()
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]
        gen_cost = time.time() - begin
        print(f"Time cost of generation: {gen_cost}s")
        print(outputs)

        return outputs
