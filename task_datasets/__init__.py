from .ocr_datasets import ocrDataset
from .vqa_datasets import TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset, ScienceQADataset, WHOOPSVQADataset
from .caption_datasets import NoCapsDataset, FlickrDataset, WHOOPSCaptionDataset
from .kie_datasets import SROIEDataset
from .imagenet_datasets import ImageNetDataset
from .cifar_datasets import CIFAR10Dataset, CIFAR100Dataset
from .oxford_iiit_pet import OxfordIIITPet
from .flowers102 import Flowers102
from .whoops_weird_dataset import WHOOPSWeirdDataset


dataset_class_dict = {
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCR-VQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'SROIE': SROIEDataset,
    'ImageNet': ImageNetDataset,
    'CIFAR10': CIFAR10Dataset,
    'CIFAR100': CIFAR100Dataset,
    'OxfordIIITPet': OxfordIIITPet,
    'Flowers102': Flowers102,
    'WHOOPSCaption': WHOOPSCaptionDataset,
    'WHOOPSVQA': WHOOPSVQADataset,
    'WHOOPSWeird': WHOOPSWeirdDataset,
}
