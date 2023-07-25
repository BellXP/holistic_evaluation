import os
import json
import argparse
import datetime

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_MRR, evaluate_embodied, evaluate_zero_shot_image_classification
from tiny_datasets import dataset_class_dict, GeneralDataset
from models import get_model


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
    parser.add_argument("--eval_binary", action="store_true", help="Whether to evaluate on binary choices.")
    parser.add_argument("--eval_multi", action="store_true", help="Whether to evaluate on multiple choices.")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
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
    if args.eval_ocr:
        return evaluate_OCR, 'VQA'
    if args.eval_vqa:
        return evaluate_VQA, 'VQA'
    if args.eval_caption:
        return evaluate_Caption, 'Caption'
    if args.eval_kie:
        return evaluate_KIE, 'VQA'
    if args.eval_mrr:
        return evaluate_MRR, 'VQA'
    if args.eval_embod:
        return evaluate_embodied, 'Embodied'
    if args.eval_cls:
        return evaluate_zero_shot_image_classification, 'VQA'
    if args.eval_binary:
        return evaluate_VQA, 'Binary'
    if args.eval_multi:
        return evaluate_VQA, 'Multi'

    return None


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    eval_function, task_type = get_eval_function(args)
    if eval_function is not None:
        # dataset = dataset_class_dict[args.dataset_name]()
        # dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        dataset = GeneralDataset(args.dataset_name)
        metrics = eval_function(model, dataset, args.model_name, args.dataset_name, task_type, time, args.batch_size, answer_path=answer_path)
        result[args.dataset_name] = metrics

    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)