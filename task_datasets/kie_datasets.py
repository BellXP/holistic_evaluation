import os
import re
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


# NOTE: the following code process KIE task like VQA

class VQA_SROIEDataset(Dataset):
    def __init__(
        self,
        dir_path= "./data/SROIE",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".txt") and '(' not in file_name:
                file_path = os.path.join(dir_path, file_name)
                img_path = file_path.replace('.txt', '.jpg')
                with open(file_path) as f:
                    content = f.read()
                    info = json.loads(content)
                    if 'company' in info.keys():
                        self.question_list.append("what is the name of the company that issued this invoice?")#llava 0.12
                        #self.question_list.append("what is the company information in the image?")#llava 0.08
                        self.answer_list.append(info['company'])
                        self.image_list.append(img_path)
                    if 'date' in info.keys():
                        self.question_list.append("when was this invoice issued?")
                        #self.question_list.append("what is the date information in the image?")
                        self.answer_list.append(info['date'])
                        self.image_list.append(img_path)

                    if 'address' in info.keys():
                        self.question_list.append("where was this invoice issued?")
                        #self.question_list.append("what is the address information in the image?")
                        self.answer_list.append(info['address'])
                        self.image_list.append(img_path)

                    if 'total' in info.keys():
                        self.question_list.append("what is the total amount of this invoice?")
                        #self.question_list.append("what is the total information in the image?")
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
    def __init__(self, ann_dir_path= "./data/FUNSD/testing_data/annotations"):
        questions = []
        answers = []
        images = []
        for file_name in os.listdir(ann_dir_path):
            file_path = os.path.join(ann_dir_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)['form']
                #去除空的linking
                json_data = [d for d in json_data if "linking" in d and len(d["linking"])>0]
                question_list = [d for d in json_data if d.get('label') == 'question']
                answer_list = [d for d in json_data if d.get('label') == 'answer']
                
                for i in range(len(question_list)):
                    link = question_list[i]['linking']
                    gt_answer = ""
                    for j in range(len(link)):
                        for k in range(len(answer_list)):
                            if answer_list[k]['id'] == link[j][1]:
                                if len(gt_answer)>0:
                                    gt_answer = gt_answer + ' ' + answer_list[k]['text']
                                else:
                                    gt_answer = gt_answer + answer_list[k]['text']
                    if len(gt_answer)>0:
                        questions.append(f"what is \"{question_list[i]['text']}\" information in the image?")
                        answers.append(gt_answer)
                        images.append(file_path.replace('annotations','images').replace('.json','.png'))
        self.questions = questions
        self.answers = answers
        self.images = images
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
    

entities = {"CE-PS":"Calories/Energy of per serving", "TF-PS":"Total fat of per serving", "CAR-PS":"Total carbohydrate of per serving",
             "PRO-PS":"Protein of per serving","SS":"Serving size", "SO-PS":"Sodium of per serving", "TF-D":"Total fat of daily value",
             "CAR-D":"Total carbohydrate of daily value","SO-D":"Sodium of daily value", "CE-P1":"Calories/Energy of per 100g/ml",
             "PRO-P1":"Protein of per 100g/ml","CAR-P1":"Total carbohydrate of per 100g/ml","TF-P1":"Total Fat of per 100g/ml", 
             "PRO-D":"Protein of daily value","SO-P1":"Sodium of per 100g/ml", "CE-D":"Calories/Energy of daily value",
            "TF-PP":"Total fat of per 100g/ml percentage","CAR-PP":"Total carbohydrate of per 100g/ml percentage", 
            "SO-PP":"Sodium of per 100g/ml percentage","PRO-PP":"Protein of per 100g/ml percentage",
            "CE-PP":"Calories/Energy of per 100g/ml percentage"}


class POIEDataset(Dataset):
    def __init__(
        self,
        dir_path= "./data/POIE/test.txt",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        with open(dir_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dict = json.loads(line)
                for key, value in dict['entity_dict'].items():
                    self.image_list.append(dir_path.replace("test.txt", dict['file_name']))
                    self.question_list.append(f'what is {entities[key]} in the image?')
                    matches = re.findall(r"\((.*?)\)", value)
                    answer = [match.strip() for match in matches]
                    answer.append(re.sub(r'\(.*?\)', '', value).strip())
                    self.answer_list.append(answer)
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


if __name__ == "__main__":
    data = POIEDataset("/home/zhangli/GPT4/MutimodelOCR/data/POIE/test.txt")
    data = iter(data)
    batch = next(data)
