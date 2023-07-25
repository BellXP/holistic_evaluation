import os
import json
import shutil
import pickle
import argparse

import torch
import numpy as np
from PIL import Image

from tiny_datasets import dataset_class_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)

    # datasets
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=20230719)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")

    # eval choices
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_mrr", action="store_true", default=False, help="Whether to evaluate on mrr.")
    parser.add_argument("--eval_embod", action="store_true", default=False, help="Whether to evaluate on embodied.")
    parser.add_argument("--eval_cls", action="store_true", default=False, help="Whether to evaluate on zero-shot classification.")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, dataset_name, max_sample_num=5000, seed=0):
    sampled_indices = 'bard_RandomSeed20230719_50samples_per_dataset.jsonl'
    with open(sampled_indices, 'r') as f:
        dataset_indices = [json.loads(x) for x in f.readlines()]
    new_dataset_indices = {x['dataset']: x['indices'] for x in dataset_indices}
    if dataset_name in new_dataset_indices:
        selected_indices = new_dataset_indices[dataset_name]

        max_sample_num = max(selected_indices) + 1
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )

        dataset = torch.utils.data.Subset(dataset, random_indices[selected_indices])
    new_dataset = []
    new_dataset_path = f'tiny_lvlm_datasets/{dataset_name}'
    os.makedirs(new_dataset_path, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        old_image_path = sample['image_path']
        if type(old_image_path) is str:
            image_name = old_image_path.split('/')[-1]
            new_image_path = f"{new_dataset_path}/{image_name}"
            shutil.copy(old_image_path, new_image_path)
        else:
            image = Image.fromarray(old_image_path)
            new_image_path = f"{new_dataset_path}/{i:02d}.png"
            image.save(new_image_path)
        sample['image_path'] = new_image_path
        new_dataset.append(sample)
    with open(f"{new_dataset_path}/dataset.pkl", 'wb') as f:
        pickle.dump(new_dataset, f)

    return dataset


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    dataset = dataset_class_dict[args.dataset_name]()
    dataset = sample_dataset(dataset, args.dataset_name, args.sample_num, args.sample_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)