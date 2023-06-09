DATA_DIR = '/nvme/share/datasets'

from .ocr_datasets import ocrDataset
from .caption_datasets import NoCapsDataset, FlickrDataset
from .kie_datasets import SROIEDataset, FUNSDDataset, POIEDataset
from .vqa_datasets import (
    TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset,
    ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset,
    VQAv2Dataset, VQAv1Dataset, VisdialDataset, IconQADataset,
    VSRDataset
)
from .embod_datasets import EmbodiedDataset

dataset_class_dict = {
    # Caption Datasets
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
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
    'OKVQA': OKVQADataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset,
    'VQAv1': VQAv1Dataset,
    'Visdial': VisdialDataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    # Embodied Datasets
    "MetaWorld": EmbodiedDataset,
    "FrankaKitchen": EmbodiedDataset,
    "Minecraft": EmbodiedDataset,
    "VirtualHome": EmbodiedDataset,
    "MinecraftPolicy": EmbodiedDataset,
}
