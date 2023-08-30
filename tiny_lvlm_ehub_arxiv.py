import os
import json
import argparse
import datetime

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from typing import Optional

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_zero_shot_image_classification
from task_datasets import ocrDataset, dataset_class_dict
# from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--question", type=str, default='The photo of the')
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--per_class_acc", action="store_true", help="mean per-class accuracy")
    
    # datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP CUTE80 COCO-Text Total-Text WordArt CTW HOST WOST")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=20230719)
    parser.add_argument("--sample_tiny", action="store_true", help="sample from Tiny LVLM-eHub")
    parser.add_argument(
        "--tiny_dataset_indices",
        default='datasets/tiny_lvlm_ehub_random_50samples/bard_RandomSeed20230719_50samples_per_dataset.jsonl',
        type=str, help="Tiny LVLM-eHub dataset indices"
    )

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_0shot_cls", action="store_true", default=False, help="Whether to evaluate on kie.")

    args = parser.parse_args()
    return args


def sample_dataset(
    dataset, max_sample_num=5000, seed=0,
    tiny: bool=False, indices: Optional[list]=None
):
    if tiny:
        max_ind = max(indices)
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_ind+1, replace=False
        )
        inds = random_indices[indices]
        dataset = torch.utils.data.Subset(dataset, inds)
        return dataset

    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def get_eval_function(args):
    if args.eval_vqa:
        return evaluate_VQA
    if args.eval_caption:
        return evaluate_Caption
    if args.eval_kie:
        return evaluate_KIE
    if args.eval_0shot_cls:
        return evaluate_zero_shot_image_classification
    return None


def main(args):
    # model = get_model(args.model_name, device=torch.device('cpu' if args.device == -1 else f"cuda:{args.device}"))
    # time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # answer_path = f"{args.answer_path}/{args.model_name}/{args.dataset_name}"
    root_dir = 'datasets/tiny_lvlm_ehub/samples_jsonl'
    Path(root_dir).mkdir(exist_ok=True)
    dataset_indices = {}
    if args.tiny_dataset_indices:
        with open(args.tiny_dataset_indices, 'r') as f:
            for line in f.readlines():
                a = json.loads(line)
                dataset = a['dataset']
                indices = a['indices']
                dataset_indices[dataset] = indices
    # result = {}
    # if args.eval_ocr:
    # OCR_datasets
    task = 'OCR'
    ocr_dataset_name = args.ocr_dataset_name.split()
    for i in range(len(ocr_dataset_name)):
        dataset_name = ocr_dataset_name[i]
        dataset = ocrDataset(dataset_name)
        dataset = sample_dataset(
            dataset, args.sample_num, args.sample_seed, tiny=args.sample_tiny,
            indices=dataset_indices[dataset_name] if args.sample_tiny else None
        )
        jsonl_path = f'{root_dir}/{task}_{dataset_name}_RandomSeed{args.sample_seed}_{args.sample_num}samples.jsonl'
        with open(jsonl_path, 'w') as f:
            for x in dataset:
                f.write(json.dumps(x) + '\n')

            # metrics = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time, batch_size=args.batch_size, answer_path=answer_path)
            # result[ocr_dataset_name[i]] = metrics

    # eval_function = get_eval_function(args)
    # if eval_function is not None:

    # use 'ScienceQAIMG' instead of 'ScienceQA'
    task_datasets = {
        'CLS': ['Flowers102', 'OxfordIIITPet', 'ImageNet',
                # 'ImageNetC_blur', 'ImageNetC_digital', 'ImageNetC_noise',
                # 'ImageNetC_weather', 'ImageNetC_extra'
        ],
        # 'Caption': [
        #     'Flickr', 'NoCaps', 'WHOOPSCaption','MSCOCO_caption_karpathy'
        # ],
        'KIE': [
            'SROIE', 'FUNSD',
            # 'POIE'
        ],
        'VQA': [
            'DocVQA', 'GQA', 'IconQA', 'OCRVQA', 'STVQA', 'TextVQA', 
            'VSR', 'OKVQA', 'WHOOPSVQA', 'VizWiz', 'ScienceQAIMG',
            # 'WHOOPSWeird', 'VQAv2', 'Visdial',
            # 'AOKVQAOpen', 'AOKVQAClose', 'HatefulMemes', 
            'ImageNetVC_shape', 'ImageNetVC_color', 'ImageNetVC_component',
            'ImageNetVC_material', 'ImageNetVC_others',
            # 'RSVQALR_mci', 'COD10K'
        ],
        'OC': ['MSCOCO_OC', 'VCR1_OC'],
        'MCI': ['MSCOCO_MCI', 'VCR1_MCI'],
        'POPE': ['MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial'],
    }
    for task in task_datasets:
        for dataset_name in task_datasets[task]:
            dataset = dataset_class_dict[dataset_name]()
            dataset = sample_dataset(
                dataset, args.sample_num, args.sample_seed, tiny=args.sample_tiny,
                indices=dataset_indices[dataset_name] if args.sample_tiny else None
            )
            if dataset_name == 'ScienceQAIMG':
                dataset_name = 'ScienceQA'
            jsonl_path = f'{root_dir}/{task}_{dataset_name}_RandomSeed{args.sample_seed}_{args.sample_num}samples.jsonl'
            with open(jsonl_path, 'w') as f:
                for x in dataset:
                    f.write(json.dumps(x) + '\n')
    cifar = {
        'CIFAR10': 'datasets/CLS_Datasets/cifar-10-batches-py/images',
        # 'CIFAR100': 'datasets/cifar-100-python/images'
    }
    task = 'CLS'
    for dataset_name in cifar:
        dataset = dataset_class_dict[dataset_name]()
        dataset = sample_dataset(
            dataset, args.sample_num, args.sample_seed, tiny=args.sample_tiny,
            indices=dataset_indices[dataset_name] if args.sample_tiny else None
        )
        jsonl_path = f'{root_dir}/{task}_{dataset_name}_RandomSeed{args.sample_seed}_{args.sample_num}samples.jsonl'
        with open(jsonl_path, 'w') as f:
            for i, x in enumerate(dataset):
                image = Image.fromarray(x['image_path'])
                image_dir = cifar[dataset_name]
                image_path = f'{image_dir}/test_{i}.png'
                # if not Path(image_path).exists():
                #     image.save(image_path)
                x['image_path'] = image_path
                f.write(json.dumps(x) + '\n')

    root_dir = Path(root_dir)
    cifar10_test = Path('datasets/cifar-10-batches-py/test_batch')
    cifar10_test_dst = root_dir.parent / 'datasets/CLS_Datasets/cifar-10-batches-py/test_batch'
    shutil.copyfile(cifar10_test, cifar10_test_dst)
    all_jsonls = list(root_dir.glob('*.jsonl'))
    for x in all_jsonls:
        if 'CIFAR10' in str(x):
            continue
        with open(x, 'r') as f:
            bar = [json.loads(y) for y in f.readlines()]
            for b in bar:
                a = Path(b['image_path'])
                new_dir = root_dir.parent / a.parent
                os.makedirs(new_dir, exist_ok=True)
                if not (new_dir / a.name).exists():
                    shutil.copy2(a, new_dir)

        # metrics = eval_function(
        #     model, dataset, args.model_name, args.dataset_name, time,
        #     batch_size=args.batch_size, answer_path=answer_path,
        #     question=args.question,
        #     max_new_tokens=args.max_new_tokens,
        #     prompt_template=args.prompt_template,
        #     per_class_acc=args.per_class_acc,
        #     )
        # result[args.dataset_name] = metrics
    
    # result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    # with open(result_path, "w") as f:
    #     f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
