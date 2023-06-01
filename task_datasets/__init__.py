from .ocr_datasets import ocrDataset
from .vqa_datasets import TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset, ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset, VQAv2Dataset
from .caption_datasets import NoCapsDataset, FlickrDataset
from .kie_datasets import SROIEDataset


dataset_class_dict = {
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCRVQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'SROIE': SROIEDataset,
    'OKVQA': OKVQADataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset
}
