import os
import json
from torch.utils.data import Dataset

import requests
from PIL import Image


class NoCapsDataset(Dataset):
    data_root = 'datasets/NoCaps'

    def __init__(self):
        self.image_list = []
        self.answer_list = []
        dataset = self.prepare_dataset()
        for img_id in dataset:
            sample_info = dataset[img_id]
            image_path = sample_info.pop(0)
            self.image_list.append(image_path)
            self.answer_list.append(sample_info)

    def prepare_dataset(self):
        dataset_file = os.path.join(self.data_root, 'val_dataset.json')
        if os.path.exists(dataset_file):
            dataset = json.load(open(dataset_file, 'r'))
        else:
            data_file = os.path.join(self.data_root, 'nocaps_val_4500_captions.json')
            data = json.load(open(data_file, 'r'))
            dataset = {}
            from tqdm import tqdm
            for sample in tqdm(data['images']):
                file_name = self.data_root + '/val_imgs/' + sample['file_name']
                img_url = sample['coco_url']
                img_id = sample['id']
                dataset[img_id] = [file_name]
                try:
                    image = Image.open(file_name).convert('RGB')
                except Exception:
                    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                    image.save(file_name)
            
            for sample in data['annotations']:
                img_id = sample['image_id']
                caption = sample['caption']
                dataset[img_id].append(caption)
            
            with open(dataset_file, 'w') as f:
                f.write(json.dumps(dataset, indent=4))
        
        return dataset

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return {
            "image_path": f'{self.data_root}/{self.image_list[idx]}',
            "gt_answers": self.answer_list[idx]}


class FlickrDataset(Dataset):
    data_root = 'datasets/Flickr_30k'

    def __init__(self):
        self.image_list = []
        self.answer_list = []
        dataset = self.prepare_dataset()
        for img_name in dataset:
            sample_info = dataset[img_name]
            image_path = f'{self.data_root}/flickr30k-images/{img_name}'
            self.image_list.append(image_path)
            self.answer_list.append(sample_info)

    def prepare_dataset(self):
        dataset_file = os.path.join(self.data_root, 'dataset.json')
        if os.path.exists(dataset_file):
            dataset = json.load(open(dataset_file, 'r'))
        else:
            data_file = os.path.join(self.data_root, 'results_20130124.token')
            dataset = {}
            with open(data_file, 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    assert len(line) == 2
                    img_name = line[0][:-2]
                    caption = line[1]
                    if img_name not in dataset:
                        dataset[img_name] = []
                    dataset[img_name].append(caption)
            
            with open(dataset_file, 'w') as f:
                f.write(json.dumps(dataset, indent=4))
        
        return dataset

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return {
            "image_path": self.image_list[idx],
            "gt_answers": self.answer_list[idx]}

class WHOOPSCaptionDataset(Dataset):
    def __init__(
        self,
        root: str='datasets/whoops',
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = f'{root}/whoops_images'
        self.anno_path = f'{root}/whoops_captions.json'
        self.annotation = json.load(open(self.anno_path, "r"))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index: int):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        answers = ann['caption']

        return {
            "image_path": image_path,
            "gt_answers": answers,
        }


if __name__ == "__main__":
    dataset = FlickrDataset()