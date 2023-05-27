from .ocr_dataset import ocrDataset
from .vqa_dataset import textVQADataset, docVQADataset, ocrVQADataset, STVQADataset, ScienceQADataset
from .caption_datasets import NoCapsDataset, FlickrDataset


dataset_class_dict = {
    'TextVQA': textVQADataset,
    'DocVQA': docVQADataset,
    'OCR-VQA': ocrVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset
}
