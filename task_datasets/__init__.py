DATA_DIR = 'datasets' # '.', '/nvme/share/leimeng/datasets'

import os
import pickle
from functools import partial
from torch.utils.data import Dataset

from .ocr_datasets import ocrDataset
from .caption_datasets import NoCapsDataset, FlickrDataset, COCOCaptionDataset, COCOCaptionKarpathyDataset
from .kie_datasets import SROIEDataset, FUNSDDataset, POIEDataset
from .embod_datasets import EmbodiedDataset
from .cls_datasets import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, OxfordIIITPet, Flowers102, ImageNetC
from .whoops import WHOOPSCaptionDataset, WHOOPSVQADataset, WHOOPSWeirdDataset
from .vqa_datasets import (
    TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset,
    ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset,
    VQAv2Dataset, VQAv1Dataset, VisdialDataset, IconQADataset,
    VSRDataset, VCR1_MCIDataset, VCR1_OCDataset, MSCOCO_MCIDataset,
    MSCOCO_OCDataset, MSCOCO_POPEDataset, MSCOCO_POPEDataset_adversarial,
    MSCOCO_POPEDataset_popular, AOKVQAOpenDataset, AOKVQACloseDataset,
    HatefulMemes, ScienceQAIMGDataset, ImageNetVC, RSVQALR, COD10K,
)


class GeneralDataset(Dataset):
    def __init__(
        self,
        dataset_name
    ):
        self.dataset_name = dataset_name
        # self.dataset = pickle.load(open(f"{DATA_DIR}/tiny_lvlm_datasets/{dataset_name}/dataset.pkl", 'rb'))
        self.dataset = pickle.load(open(f"tiny_lvlm_datasets/{dataset_name}/dataset.pkl", 'rb'))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # sample['image_path'] = f"{DATA_DIR}/{sample['image_path']}"
        sample['image_path'] = f"{sample['image_path']}"
        return sample


dataset_class_dict = {
    # Caption Datasets
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'MSCOCO_caption': COCOCaptionDataset,
    'MSCOCO_caption_karpathy': COCOCaptionKarpathyDataset,
    # KIE Datasets
    'SROIE': SROIEDataset,
    'FUNSD': FUNSDDataset,
    'POIE': POIEDataset,
    # VQA Datasets
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCRVQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'ScienceQAIMG': ScienceQAIMGDataset,
    'OKVQA': OKVQADataset,
    'AOKVQAOpen': AOKVQAOpenDataset,
    'AOKVQAClose': AOKVQACloseDataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset,
    'VQAv1': VQAv1Dataset,
    'Visdial': VisdialDataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    'HatefulMemes': HatefulMemes,
    'ImageNetVC_color': partial(ImageNetVC, task='color'),
    'ImageNetVC_component': partial(ImageNetVC, task='component'),
    'ImageNetVC_material': partial(ImageNetVC, task='material'),
    'ImageNetVC_others': partial(ImageNetVC, task='others'),
    'ImageNetVC_shape': partial(ImageNetVC, task='shape'),
    # Embodied Datasets
    "MetaWorld": partial(EmbodiedDataset, dataset_name="MetaWorld"),
    "FrankaKitchen": partial(EmbodiedDataset, dataset_name="FrankaKitchen"),
    "Minecraft": partial(EmbodiedDataset, dataset_name="Minecraft"),
    "VirtualHome": partial(EmbodiedDataset, dataset_name="VirtualHome"),
    "MinecraftPolicy": partial(EmbodiedDataset, dataset_name="MinecraftPolicy"),
    # classification
    'ImageNet': ImageNetDataset,
    'CIFAR10': CIFAR10Dataset,
    'CIFAR100': CIFAR100Dataset,
    'OxfordIIITPet': OxfordIIITPet,
    'Flowers102': Flowers102,
    'ImageNetC': ImageNetC,
    'ImageNetC_blur': partial(ImageNetC, mode='blur'),
    'ImageNetC_digital': partial(ImageNetC, mode='digital'),
    'ImageNetC_noise': partial(ImageNetC, mode='noise'),
    'ImageNetC_weather': partial(ImageNetC, mode='weather'),
    'ImageNetC_extra': partial(ImageNetC, mode='extra'),
    # whoops
    'WHOOPSCaption': WHOOPSCaptionDataset,
    'WHOOPSVQA': WHOOPSVQADataset,
    'WHOOPSWeird': WHOOPSWeirdDataset,
    # OC, MCI, Hallucination
    'VCR1_OC': VCR1_OCDataset,
    'VCR1_MCI': VCR1_MCIDataset,
    'MSCOCO_MCI': MSCOCO_MCIDataset,
    'MSCOCO_OC': MSCOCO_OCDataset,
    'MSCOCO_pope_random': MSCOCO_POPEDataset,
    'MSCOCO_pope_popular': MSCOCO_POPEDataset_popular,
    'MSCOCO_pope_adversarial': MSCOCO_POPEDataset_adversarial,
    'RSVQALR_MCI': partial(RSVQALR, q_type='presence'),
    'COD10K': COD10K,
    # OCR
    "COCO-Text": partial(ocrDataset, dataset_name="COCO-Text"),
    "CTW": partial(ocrDataset, dataset_name="CTW"),
    "CUTE80": partial(ocrDataset, dataset_name="CUTE80"),
    "HOST": partial(ocrDataset, dataset_name="HOST"),
    "IC13": partial(ocrDataset, dataset_name="IC13"),
    "IC15": partial(ocrDataset, dataset_name="IC15"),
    "IIIT5K": partial(ocrDataset, dataset_name="IIIT5K"),
    "SVTP": partial(ocrDataset, dataset_name="SVTP"),
    "SVT": partial(ocrDataset, dataset_name="SVT"),
    "Total-Text": partial(ocrDataset, dataset_name="Total-Text"),
    "WOST": partial(ocrDataset, dataset_name="WOST"),
    "WordArt": partial(ocrDataset, dataset_name="WordArt"),
}
