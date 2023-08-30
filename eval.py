import os
import json
import argparse
import datetime

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_zero_shot_image_classification
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model
from typing import Optional

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
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--tiny_indices_jsonl", type=str, default=None,
        help="Tiny LVLM-eHub dataset indices jsonl file path."
    )

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_0shot_cls", action="store_true", default=False, help="Whether to evaluate on kie.")

    parser.add_argument(
        "--ablate_prompts", action="store_true", default=False,
        help="Ablate prompts for zero-shot image classification."
    )

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0, indices: Optional[list]=None):
    if indices:
        max_sample_num = max(indices) + 1
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        if indices:
            random_indices = random_indices[indices]
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset

def sample_imagenet1k(dataset, max_sample_num=3, seed=0):
    num_classes = 1000
    num_samples_per_class = 50
    inds = []
    np.random.seed(seed)
    for i in range(num_classes):
        inds_i = np.random.choice(
            num_samples_per_class, max_sample_num, replace=False
        ) + i * num_samples_per_class
        inds.extend(inds_i.tolist())
    dataset = torch.utils.data.Subset(dataset, inds)
    # print(f'sample ImageNet1K for prompt analysis {len(inds)}')
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
    model = get_model(args.model_name, device=torch.device('cpu' if args.device == -1 else f"cuda:{args.device}"))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}/{args.dataset_name}"
    if not os.path.exists(answer_path):
        os.makedirs(answer_path)

    if args.tiny_indices_jsonl:
        dataset2indices = {}
        with open(args.tiny_indices_jsonl, 'r') as f:
            for line in f.readlines():
                a = json.loads(line)
                dataset = a['dataset']
                indices = a['indices']
                dataset2indices[dataset] = indices

    result = {}
    if args.eval_ocr:
        ocr_dataset_name = args.ocr_dataset_name.split()
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(ocr_dataset_name[i])
            dataset = sample_dataset(
                dataset, args.sample_num, args.sample_seed,
                dataset2indices[ocr_dataset_name[i]] if args.tiny_indices_jsonl else None
            )
            metrics = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time, batch_size=args.batch_size, answer_path=answer_path)
            result[ocr_dataset_name[i]] = metrics

    eval_function = get_eval_function(args)
    if eval_function is not None:
        dataset = dataset_class_dict[args.dataset_name]()
        if args.dataset_name == 'ImageNet' and args.ablate_prompts:
            dataset = sample_imagenet1k(dataset, args.sample_num, args.sample_seed)
        else:
            dataset = sample_dataset(
                dataset, args.sample_num, args.sample_seed,
                dataset2indices[args.dataset_name] if args.tiny_indices_jsonl else None
            )
        metrics = eval_function(
            model, dataset, args.model_name, args.dataset_name, time,
            batch_size=args.batch_size, answer_path=answer_path,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            prompt_template=args.prompt_template,
            per_class_acc=args.per_class_acc,
        )
        result[args.dataset_name] = metrics
    
    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
