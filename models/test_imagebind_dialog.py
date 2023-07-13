from gradio_client import Client
from . import get_image, DATA_DIR
from . import llama_imagebind as llama
from .llama_imagebind.ImageBind import data


class TestImageBind_Dialog:
    def __init__(self) -> None:
        llama_dir = f'{DATA_DIR}/llama_checkpoints'
        self.model = llama.load("ckpts/checkpoint-from7B.pth", llama_dir, knn=True, llama_type="7B_chinese", max_seq_len=128, max_batch_size=16)
        self.model.eval()

    def generate(self, image, question, max_new_tokens=128):
        results = self.model.generate(
            {"Image": [data.load_and_transform_vision_data([image], device='cuda'), 1]},
            [llama.format_prompt(question)],
            max_gen_len=max_new_tokens
        )
        result = results[0].strip()
        return result

    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        results = self.model.generate(
            {"Image": [data.load_and_transform_vision_data(image_list, device='cuda'), 1]},
            [llama.format_prompt(question) for question in question_list],
            max_gen_len=max_new_tokens, temperature=0,
        )
        return results