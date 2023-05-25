from PIL import Image
from gradio_client import Client


class Web_Model_Worker:
    def __init__(self) -> None:
        self.model = Client("http://106.14.127.192:8088/")
        self.max_length = 64
        self.temperature = 0.1
        self.top_p = 0.75

    def generate(self, image, question: str):
        if type(image) is str:
            image_name = image
        elif type(image) is Image.Image:
            image_name = '.llama_adapter_v2_inference.png'
            image.save(image_name)
        else:
            raise NotImplementedError

        output = self.model.predict(image_name, question, self.max_length, self.temperature, self.top_p, fn_index=1)
        
        return output
    