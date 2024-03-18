import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

from . import DATA_DIR


class TestLLM:
    def __init__(self, model_name, device=None) -> None:
        device = 'cuda' if device is None else device
        self.use_caption = 'Caption' in model_name
        if 'LLaMA-7B' in model_name:
            self.model_type = 'LLaMA'
            model_id = 'luodian/llama-7b-hf'
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
            self.model.eval()
        elif 'Vicuna-7B' in model_name:
            self.model_type = 'Vicuna'
            # model_id = 'luodian/vicuna-7b-hf'
            model_id = f"{DATA_DIR}/PandaGPT/vicuna_ckpt/7b_v0"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
        elif 'Flan-T5-XL' in model_name:
            self.model_type = 'Flan-T5'
            model_id = 'google/flan-t5-xl'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.model_type != 'Flan-T5':
            self.tokenizer.pad_token='[PAD]'

        if self.use_caption:
            self.caption_dict = json.load(open('ImageNetVC_caption_dict.json')) | json.load(open('VCR_val_caption_dict.json'))        

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=1024):
        assert type(image) is str
        image = image.replace('/mnt/lustre/xupeng/datasets/', '')
        if self.use_caption and image in self.caption_dict:
            image_caption = self.caption_dict[image]
            prompt = f"Image Description: {image_caption}\n\n{question}"
        else:
            prompt = question

        if self.use_flan:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        else:
            response = self.generate_text(
                prompt,
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=64,
            )[0]['generated_text'][len(prompt):].strip()

        return response

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        prompt_list = []
        for image, question in zip(image_list, question_list):
            assert type(image) is str
            image = image.replace('/mnt/lustre/xupeng/datasets/', '')
            if self.use_caption and image in self.caption_dict:
                image_caption = self.caption_dict[image]
                prompt = f"Image Description: {image_caption}\n\n{question}"
            else:
                prompt = question
            if self.model_type == 'Vicuna':
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: {prompt}\n### Assistant:"
            prompt_list.append(prompt)

        inputs = self.tokenizer(prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        max_prompt_length = inputs['input_ids'].shape[1]

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens + max_prompt_length, do_sample=False)
        responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        if self.model_type != 'Flan-T5':
            responses = [x[len(prompt):].strip() for x, prompt in zip(responses, prompt_list)]
        # import pdb; pdb.set_trace()

        return responses