import os
import glob
import torch
import torchvision.transforms as transforms

from .meta import MetaModel
from .utils import format_prompt


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=None), # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


transform_val = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def load(model_path, llama_type, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", max_seq_len=512, max_batch_size=1):
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_config_path = os.path.join(llama_ckpt_dir, 'params.json')
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    model = MetaModel('llama_qformerv2_peft', llama_config_path, llama_tokenzier_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    ckpts = sorted(glob.glob(f"{model_path}/consolidated*.model.pth"))
    for ckpt in ckpts:
        ckpt = torch.load(ckpt, map_location='cpu')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)

    return model.to(device), transform_train
