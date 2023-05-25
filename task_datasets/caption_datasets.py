import os
import json
from torch.utils.data import Dataset

import requests
from PIL import Image


class NoCapsDataset(Dataset):
    data_root = '/nvme/share/Caption_Datasets/NoCaps'

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
                image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                image.save(file_name)
            
            for sample in data['annotations']:
                img_id = sample['image_id']
                caption = sample['caption']
                dataset[img_id].append(caption)
        
        return dataset

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return {
            "image_path": self.image_list[idx],
            "gt_answers": self.answer_list[idx]}