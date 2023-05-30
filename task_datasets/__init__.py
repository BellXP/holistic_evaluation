from .ocr_datasets import ocrDataset
from .vqa_datasets import TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset, ScienceQADataset
from .caption_datasets import NoCapsDataset, FlickrDataset
from .kie_datasets import SROIEDataset


dataset_class_dict = {
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCR-VQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'SROIE': SROIEDataset
}
