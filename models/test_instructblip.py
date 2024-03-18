import torch
import contextlib
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from . import get_image


# No sampling in default
class TestInstructBLIP:
    def __init__(self, base_llm='vicuna', device=None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if base_llm == 'vicuna':
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.model.to(self.device, dtype=torch.bfloat16)
        elif base_llm == 't5':
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            self.model.to(self.device)
        else:
            raise NotImplementedError(f"Invalid base llm: {base_llm}")
        self.inference_context = contextlib.nullcontext() if base_llm == 't5' else torch.cuda.amp.autocast()

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=1024, do_sample=False, num_beams=1):
        image = get_image(image)
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        with self.inference_context:
            outputs = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_length=max_new_tokens,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1024, do_sample=False, num_beams=1):
        images = [get_image(image) for image in image_list]
        inputs = self.processor(images=images, text=question_list, return_tensors="pt", padding=True).to(self.device)
        with self.inference_context:
            outputs = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_length=max_new_tokens,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return generated_text