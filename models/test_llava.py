import torch
from threading import Thread
from transformers import TextIteratorStreamer

from . import get_image, DATA_DIR
from .llava.model.builder import load_pretrained_model
from .llava.conversation import conv_templates, SeparatorStyle
from .llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def get_template_name(model_name):
    if "llava" in model_name.lower():
        if 'llama-2' in model_name.lower():
            template_name = "llava_llama_2"
        elif "v1" in model_name.lower():
            if 'mmtag' in model_name.lower():
                template_name = "v1_mmtag"
            elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                template_name = "v1_mmtag"
            else:
                template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt"
        else:
            if 'mmtag' in model_name.lower():
                template_name = "v0_mmtag"
            elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                template_name = "v0_mmtag"
            else:
                template_name = "llava_v0"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "llama-2" in model_name:
        template_name = "llama_2"
    else:
        template_name = "vicuna_v1"
    return template_name


class TestLLaVA:
    def __init__(self, device=None):
        model_base = f'{DATA_DIR}/LLaVA/LLaVA-Vicuna-7B-v1.1'
        model_path = f'{DATA_DIR}/LLaVA/llava_vicuna-7b-v1.1-lcs_558k-instruct-80k_lora'
        model_name = "llava_vicuna-7b-v1.1-lcs_558k-instruct-80k_lora"
        self.template_name = get_template_name(model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name)

        self.is_multimodal = 'llava' in model_name.lower()
        self.image_process_mode = 'Crop'
        self.temperature = 0
        self.top_p = 0.7

    def generate(self, image, question, max_new_tokens=1024, do_sample=False, num_beams=1):
        image = get_image(image)
        question = question[:1536]  # Hard cut-off
        if image is not None:
            question = question[:1200]  # Hard cut-off for images
            if '<image>' not in question:
                question = question + '\n<image>'
            text = (question, image, self.image_process_mode)

        state = conv_templates[self.template_name].copy()
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)

        self.temperature = 0.2 if do_sample else 0.0

        prompt = state.get_prompt()
        images = state.get_images(return_pil=True)
        stop_str = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2
        outputs = self.do_generate(prompt, images, stop_str, self.is_multimodal, self.temperature, self.top_p, max_new_tokens)
        return outputs

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1024, do_sample=False, num_beams=1):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output

    @torch.no_grad()
    def do_generate(self, prompt, images, stop_str, is_multimodal, temperature, top_p, max_new_tokens):
        num_image_tokens = 0
        if images is not None and len(images) > 0 and is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                if type(images[0]) is str:
                    images = [load_image_from_base64(image) for image in images]
                images = process_images(images, self.image_processor, self.model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * self.model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        do_sample = True if temperature > 0.001 else False
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        max_new_tokens = min(max_new_tokens, self.context_len - input_ids.shape[-1] - num_image_tokens)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(target=self.model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            num_beams=1,
            **image_args
        ))
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
        generated_text = generated_text.strip()

        return generated_text