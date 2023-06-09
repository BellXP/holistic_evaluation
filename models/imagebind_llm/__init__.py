from .. import DATA_DIR
from ..test_imagebind import MAX_SEQ_LEN, MAX_BATCH_SIZE
from functools import partial
from torchvision import transforms
from .llama import load, format_prompt

llama_dir = f'{DATA_DIR}/llama_checkpoints'
load_model = partial(load, llama_dir=llama_dir)
image_transform = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
