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
    # datasets
    parser.add_argument("dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=20230719)

    args = parser.parse_args()
    return args


def sample_dataset(dataset, dataset_name, max_sample_num=50, seed=0):
    sampled_indices = 'bard_RandomSeed20230719_50samples_per_dataset.jsonl'
    with open(sampled_indices, 'r') as f:
        dataset_indices = [json.loads(x) for x in f.readlines()]
    new_dataset_indices = {x['dataset']: x['indices'] for x in dataset_indices}
    if dataset_name in new_dataset_indices:
    # if False:
        selected_indices = new_dataset_indices[dataset_name]

        max_sample_num = max(selected_indices) + 1
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )

        print(f'The length of {dataset_name} is {len(random_indices[selected_indices])}')

        dataset = torch.utils.data.Subset(dataset, random_indices[selected_indices])
    else:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    new_dataset = []
    new_dataset_path = f'tiny_lvlm_datasets/{dataset_name}'
    os.makedirs(new_dataset_path, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        old_image_path = sample['image_path']
        if type(old_image_path) is str:
            # image = Image.open(old_image_path).convert('RGB')
            image_name = old_image_path.split('/')[-1]
            new_image_path = f"{new_dataset_path}/{image_name}"
            if dataset_name == 'IconQA':
                new_image_path = f"{new_dataset_path}/{i:02d}.png"
            shutil.copy(old_image_path, new_image_path)
        else:
            image = Image.fromarray(old_image_path)
            new_image_path = f"{new_dataset_path}/{i:02d}.png"
            image.save(new_image_path)
        sample['image_path'] = new_image_path
        # sample['image_path'] = f"{dataset_name}/{i:02d}.png"
        new_dataset.append(sample)
    with open(f"{new_dataset_path}/dataset.pkl", 'wb') as f:
        pickle.dump(new_dataset, f)

    return dataset


def main(args):
    for dataset_name in args.dataset_name.split(','):
        dataset = dataset_class_dict[dataset_name]()
        dataset = sample_dataset(dataset, dataset_name, args.sample_num, args.sample_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)