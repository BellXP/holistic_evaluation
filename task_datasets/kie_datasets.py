import os
import json
from torch.utils.data import Dataset


class SROIEDataset(Dataset):
    data_root = '/nvme/share/KIE_Datasets/SROIE'
    question = 'Extract the campany, data, address and total information shown in this image'

    def __init__(self):
        dataset = self.prepare_dataset()
        self.image_list = dataset['image_list']
        self.answer_list = dataset['answer_list']

    def prepare_dataset(self):
        dataset_file = f"{self.data_root}/dataset.json"
        if os.path.exists(dataset_file):
            dataset = json.load(open(dataset_file, 'r'))
        else:
            image_list, answer_list = [], []
            for img_file in os.listdir(f"{self.data_root}/images"):
                image_list.append(f"{self.data_root}/images/{img_file}")
                img_id = img_file.replace('.jpg', '')
                answer_file = f"{self.data_root}/gt_answers/{img_id}.txt"
                with open(answer_file, 'r') as f:
                    answer_dict = {}
                    for line in f.readlines():
                        line = line.replace('\n', '').strip()
                        line = line.split("\": \"")
                        if len(line) == 1:
                            continue
                        key = line[0][1:]
                        value = line[1][:-1] if key == 'total' else line[1][:-2]
                        answer_dict[key] = value
                    answer = ' '.join(answer_dict.values())
                    answer_list.append(answer)
            dataset = {'image_list': image_list, 'answer_list': answer_list}
            with open(dataset_file, 'w') as f:
                f.write(json.dumps(dataset, indent=4))
        return dataset

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return {
            "image_path": self.image_list[idx],
            "gt_answers": self.answer_list[idx]}
