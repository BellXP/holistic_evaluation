import os
import json
import datasets
from torch.utils.data import Dataset


class TextVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/nvme/share/VQA_Datasets/TextVQA/train_images",
        ann_path= "/nvme/share/VQA_Datasets/TextVQA/TextVQA_0.5.1_val.json"
    ):
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, f"{self.data[idx]['image_id']}.jpg")
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class DocVQADataset(Dataset):
    data_root = '/nvme/share/VQA_Datasets/DocVQA/val'

    def __init__(self):
        ann_path = f"{self.data_root}/val_v1.0.json"
        self.data = json.load(open(ann_path, "r"))["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.data_root, self.data[idx]['image'])
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class OCRVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocrVQA/images",
        ann_path= "./data/ocrVQA/dataset.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        dataset = json.load(open(ann_path, "r"))
        import pdb;pdb.set_trace()
        for idx, data in enumerate(dataset):
            questions =  dataset[data]['questions']
            for index, question in enumerate(questions):
                image_file = os.path.join(image_dir_path, f'{data}.jpg')
                gt_answers = dataset[data]['answers'][index]
                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class STVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/nvme/share/VQA_Datasets/STVQA/train_imgs",
        ann_path= "/nvme/share/VQA_Datasets/STVQA/train_task_3.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(ann_path, "r"))['data']
        for i in range(len(data)):
            image_path = image_dir_path + '/' + data[i]['dataset'] + '/' + data[i]['file_name']
            self.image_list.append(image_path)
            self.answer_list.append(data[i]['answers'])
            self.question_list.append(data[i]['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
    

class ScienceQADataset(Dataset):
    split='test'
    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_root = '/nvme/share/VQA_Datasets/ScienceQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        ann_path = f"{self.data_root}/{self.split}_anns.json"
        if os.path.exists(ann_path):
            dataset = json.load(open(ann_path, "r"))
            for sample in dataset:
                self.image_list.append(sample['image_path'])
                self.question_list.append(sample['question'])
                self.answer_list.append(sample['answer'])
        else:
            self.load_save_dataset()
    
    def load_save_dataset(self):
        # load dataset
        data = datasets.load_dataset('derek-thomas/ScienceQA', self.split)
        for sample in data[self.split]:
            if sample['image'] is None:
                continue
            # question = f"Question: {sample['question']}\n" \
            #            f"Options: {' '.join([f'({x}) {y}' for x, y in zip(self.options, sample['choices'])])}\n"
            question = f"Question: {sample['question']}\n" \
                       f"Options: {' '.join(sample['choices'])}\n"

            self.question_list.append(question)
            self.image_list.append(sample['image'].convert('RGB'))
            self.answer_list.append(sample['choices'][sample['answer']])

        # save dataset
        dataset = []
        for i in range(len(self.image_list)):
            img_file_name = f'{self.data_root}/{self.split}_imgs/{i:04d}.png'
            if not os.path.exists(img_file_name):
                self.image_list[i].save(img_file_name)
            self.image_list[i] = img_file_name
            dataset.append({
                'answer': self.answer_list[i],
                'image_path': self.image_list[i],
                'question': self.question_list[i]
            })
        with open(f"{self.data_root}/{self.split}_anns.json", "w") as f:
            f.write(json.dumps(dataset, indent=4))

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}

class WHOOPSVQADataset(Dataset):
    def __init__(
        self,
        root: str='datasets/whoops',
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = f'{root}/whoops_images'
        self.anno_path = f'{root}/whoops_vqa_pairs.json'
        self.annotation = json.load(open(self.anno_path, "r"))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        answers = ann['reference']
        question = ann['question']

        return {
            "image_path": image_path,
            "question": question,
            "gt_answers": answers}
