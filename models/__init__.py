import torch
from PIL import Image
import numpy as np


def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        img = np.array(
            Image.open(image).convert('RGB'), dtype=np.uint8)[:, :, ::-1]
        return Image.fromarray(img)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        # TBD convert RGB to BGR HxWxC
        return Image.fromarray(image[:, :, ::-1]).convert('RGB')
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_model(model_name, device=None):
    if model_name == 'BLIP2':
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
    elif model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device)
    elif model_name == 'VPGTrans':
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'LLaVA':
        from .test_llava import TestLLaVA
        return TestLLaVA(device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2, TestLLamaAdapterV2_web
        return TestLLamaAdapterV2(device)
    elif model_name == 'Multimodal-GPT':
        from .test_multimodel_gpt import TestMultiModelGPT # Web version
        return TestMultiModelGPT(device)
    elif 'ImageBind' in model_name:
        from .test_imagebind import TestImageBind
        return TestImageBind(model_name, device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
