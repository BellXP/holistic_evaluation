import torch
from transformers import CLIPImageProcessor
from .flamingo.modeling_flamingo import FlamingoForConditionalGeneration

from models import DATA_DIR
CKPT_PATH=f'{DATA_DIR}/openflamingo-9b-hf'


class TestFlamingo:
    def __init__(self, device=None) -> None:
        self.model = FlamingoForConditionalGeneration.from_pretrained(CKPT_PATH)
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
        self.model.vision_encoder = self.model.vision_encoder.to('cpu', dtype=torch.float32)

    def generate(self, image, question):
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            # vision_x=vision_x.to(self.model.device),
            vision_x=vision_x.to('cpu'),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=256,
            num_beams=1,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        begin_label = output.lower().find('<answer>')
        end_label = output.lower().find('</answer>')
        output = output[begin_label + 8: end_label].strip()

        return output
  
    