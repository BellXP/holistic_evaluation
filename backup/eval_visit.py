import os
import json
import argparse
from tqdm import tqdm

import torch
import datasets
from torch.utils.data import Dataset, DataLoader

from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA2-Accessory")
    parser.add_argument("--batch_size", type=int, default=32)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./visit_answers")

    args = parser.parse_args()
    return args


class VisITDataset(Dataset):
    split = 'TEST'
    DATA_DIR = '/mnt/lustre/xupeng/datasets'

    def __init__(self):
        dataset_path = f"{self.DATA_DIR}/VisIT-Bench/dataset.json"
        if os.path.exists(dataset_path):
            self.dataset = json.load(open(dataset_path, 'r'))
        else:
            data = datasets.load_dataset('mlfoundations/VisIT-Bench', 'TEST')
            dataset = []
            for i, sample in enumerate(data['test']):
                image_path = f"{self.DATA_DIR}/VisIT-Bench/images/{i:04d}.png"
                sample['image'].convert('RGB').save(image_path)
                sample['image'] = image_path
                dataset.append(sample)
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        image_url = self.dataset[idx]['image_url']
        instruction = self.dataset[idx]['instruction']
        reference_output = self.dataset[idx]['reference_output']
        instruction_category = self.dataset[idx]['instruction_category']
        return {
            'image': image,
            'image_url': image_url,
            'instruction': instruction,
            'reference_output': reference_output,
            'instruction_category': instruction_category
        }


def evaluate_visit(model, model_name, batch_size, answer_path):
    os.makedirs(answer_path, exist_ok=True)
    answer_path = os.path.join(answer_path, "VisIT-Bench.json")
    
    predictions=[]
    dataset = VisITDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image'], batch['instruction'])
        for image_url, instruction, reference_output, instruction_category, output in zip(batch['image_url'], batch['instruction'], batch['reference_output'], batch['instruction_category'], outputs):
            answer_dict={'image_url': image_url, 'instruction': instruction, 'reference_output': reference_output, 'instruction_category': instruction_category, 'output': output, 'model_name': model_name}
            predictions.append(answer_dict)

    if torch.distributed.get_rank() != 0:
        return
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    answer_path = f"{args.answer_path}/{args.model_name}"
    evaluate_visit(model, args.model_name, args.batch_size, answer_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)