import gc
import torch
import numpy as np
from PIL import Image

DATA_DIR = '/mnt/lustre/share_data/xupeng/models'

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_device_name(device: torch.device):
    return f"{device.type}{'' if device.index is None else ':' + str(device.index)}"


@torch.inference_mode()
def generate_stream(model, text, image, device=None, keep_in_device=False, pure=False):
    image = np.array(image, dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    
    try:
        if pure:
            output = model.pure_generate(image, text)
        else:
            output = model.generate(image, text)
    except Exception as e:
        output = getattr(e, 'message', str(e))
    
    print(f"{'#' * 20} Model out: {output}")
    gc.collect()
    torch.cuda.empty_cache()
    yield output


def get_model(model_name, device=None):
    if model_name == 'GPT4V':
        return None
    elif model_name == 'BLIP2':
        from .test_blip2 import TestBlip2
        return TestBlip2(device)
    elif model_name == 'MiniGPT-4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'mPLUG-Owl':
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'Otter':
        from .test_otter import TestOtter
        return TestOtter(device)
    elif model_name == 'Otter-Image':
        from .test_otter_image import TestOtterImage
        return TestOtterImage(device)
    elif 'InstructBLIP' in model_name:
        from .test_instructblip import TestInstructBLIP
        if 't5' in model_name.lower():
            return TestInstructBLIP('t5', device)
        else:
            return TestInstructBLIP('vicuna', device)
    elif model_name == 'VPGTrans':
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif 'LLaVA' in model_name:
        from .test_llava import TestLLaVA
        return TestLLaVA(model_name, device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(device)
    elif 'PandaGPT' in model_name:
        from .test_pandagpt import TestPandaGPT
        return TestPandaGPT(device)
    elif 'OFv2' in model_name:
        _, version = model_name.split('_')
        from .test_OFv2 import OFv2
        return OFv2(version, device)
    elif 'LaVIN' in model_name:
        from .test_lavin import TestLaVIN
        return TestLaVIN(device)
    elif model_name == 'Lynx':
        from .test_lynx import TestLynx
        return TestLynx(device)
    elif model_name == 'Cheetah':
        from .test_cheetah import TestCheetah
        return TestCheetah(device)
    elif model_name == 'BLIVA':
        from .test_bliva import TestBLIVA
        return TestBLIVA(device)
    elif model_name == 'MIC':
        from .test_mic import TestMIC
        return TestMIC(device)
    elif model_name in [f"{a}{b}" for a in ['LLaMA-7B', 'Vicuna-7B', 'Flan-T5-XL'] for b in ['-Caption', '']]:
        from .test_llm import TestLLM
        return TestLLM(model_name)
    else:
        from .test_automodel import TestAutoModel
        return TestAutoModel(model_name, device)
