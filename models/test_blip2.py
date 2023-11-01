import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from . import get_image


# No tempearture sampling in default
class TestBlip2:
    def __init__(self, device=None) -> None:
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=1024, do_sample=False, num_beams=1):
        image = get_image(image)
        inputs = self.processor(image, question, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_length=max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        output = self.processor.decode(output[0], skip_special_tokens=True)
        return output

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1024, do_sample=False, num_beams=1):
        images = [get_image(image) for image in image_list]
        inputs = self.processor(images, question_list, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return outputs
    