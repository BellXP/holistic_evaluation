import os
import re
import json
from torch.utils.data import Dataset


class SROIEDataset(Dataset):
    data_root = '/nvme/share/KIE_Datasets/SROIE'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        for file_name in os.listdir(f"{self.data_root}/gt_answers"):
            file_path = os.path.join(self.data_root, 'gt_answers', file_name)
            img_path = os.path.join(self.data_root, 'images', file_name.replace('.txt', '.jpg'))
            with open(file_path) as f:
                content = f.read()
                info = json.loads(content)
                if 'company' in info.keys():
                    self.question_list.append("what is the name of the company that issued this invoice?")
                    self.answer_list.append(info['company'])
                    self.image_list.append(img_path)
                if 'date' in info.keys():
                    self.question_list.append("when was this invoice issued?")
                    self.answer_list.append(info['date'])
                    self.image_list.append(img_path)
                if 'address' in info.keys():
                    self.question_list.append("where was this invoice issued?")
                    self.answer_list.append(info['address'])
                    self.image_list.append(img_path)
                if 'total' in info.keys():
                    self.question_list.append("what is the total amount of this invoice?")
                    self.answer_list.append(info['total'])
                    self.image_list.append(img_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class FUNSDDataset(Dataset):
    data_root = '/nvme/share/KIE_Datasets/FUNSD'

    def __init__(self):
        self.questions = []
        self.answers = []
        self.images = []

        ann_dir = f"{self.data_root}/testing_data/annotations"
        for file_name in os.listdir(ann_dir):
            file_path = os.path.join(ann_dir, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)['form']
                json_data = [d for d in json_data if "linking" in d and len(d["linking"])>0]
                question_list = [d for d in json_data if d.get('label') == 'question']
                answer_list = [d for d in json_data if d.get('label') == 'answer']
                
                for i in range(len(question_list)):
                    link = question_list[i]['linking']
                    gt_answer = ""
                    for j in range(len(link)):
                        for k in range(len(answer_list)):
                            if answer_list[k]['id'] == link[j][1]:
                                if len(gt_answer) > 0:
                                    gt_answer = gt_answer + ' ' + answer_list[k]['text']
                                else:
                                    gt_answer = gt_answer + answer_list[k]['text']
                    
                    if len(gt_answer)>0:
                        self.questions.append(f"what is \"{question_list[i]['text']}\" information in the image?")
                        self.answers.append(gt_answer)
                        self.images.append(file_path.replace('annotations','images').replace('.json','.png'))
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        question = self.questions[idx]
        answers = self.answers[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


if __name__ == "__main__":
    dataset = FUNSDDataset()
    print(len(dataset))
    print(dataset[0])
