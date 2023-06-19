import torch
from transformers import CLIPImageProcessor
from .otter.modeling_otter import OtterForConditionalGeneration
from PIL import Image

CKPT_PATH = 'checkpoints/otter-9b-hf'


class TestOtter:
    def __init__(self, device=None) -> None:
        model_path=CKPT_PATH
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)
        # self.model.vision_encoder = self.model.vision_encoder.to('cpu', dtype=torch.float32)

    @torch.no_grad()
    def generate(self, image, question):
        image = Image.open(image).convert('RGB')
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            # vision_x=vision_x.to(self.model.device),
            vision_x=vision_x.to('cpu'),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=256,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        out_label = output.index('GPT:')
        output = ' '.join(output[out_label + 1:])
        
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=128, *args, **kwargs):

        if type(image_list[0]) is not str:
            images = [Image.fromarray(x) for x in image_list]
        else:
            images = [Image.open(img).convert('RGB') for img in image_list]
        vision_x = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(1)
        prompts = [f"<image> User: {x} GPT: <answer>" for x in question_list]
        lang_x = self.model.text_tokenizer(prompts, padding='longest',return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.device),
            attention_mask=lang_x["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        results = []
        for x in generated_text:
            output = self.model.text_tokenizer.decode(x)
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            output = ' '.join(output[out_label + 1:])
            results.append(output)
        return results
