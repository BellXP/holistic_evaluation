from gradio_client import Client
from . import get_image, DATA_DIR
from . import llama_imagebind as llama
from .llama_imagebind.ImageBind import data


class TestImageBindWeb:
    def __init__(self) -> None:
        self.model = Client("http://imagebind-llm.opengvlab.com/")
        self.cache_size = 10
        self.cache_temperature = 20
        self.cache_weight = 0.5
        self.temperature = 0.1
        self.top_p = 0.75
        self.output_type = 'Text'

    def generate(self, image, question, max_new_tokens=128):
        try_times = 0
        while try_times < 10:
            try:
                if type(image) is str:
                    image_name = image
                else:
                    image = get_image(image)
                    image_name = 'models/imagebind_examples/imagebind_inference.png'
                    image.save(image_name)
                output = self.model.predict(
                    ['Image'], image_name, 1.0, 'text', 0.0,
                    'models/imagebind_examples/yoga.mp4', 0.0,
                    'models/imagebind_examples/sea_wave.wav', 0.0,
                    'models/imagebind_examples/airplane.pt', 0.0,
                    'Question', question, self.cache_size, self.cache_temperature, self.cache_weight,
                    max_new_tokens, self.temperature, self.top_p, self.output_type, fn_index=11)[0]
                
                return output
            except:
                try_times += 1
                continue
        raise ValueError("Cannot get normal output of ImageBind")

    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        output = [self.generate(image, question, max_new_tokens=max_new_tokens) for image, question in zip(image_list, question_list)]

        return output


class TestImageBind:
    def __init__(self) -> None:
        llama_dir = f'{DATA_DIR}/llama_checkpoints'
        self.model = llama.load("7B", llama_dir, knn=True, llama_type="7B_chinese", max_seq_len=128, max_batch_size=16)
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