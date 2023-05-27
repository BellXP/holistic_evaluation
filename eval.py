import os
import json
import argparse
import datetime

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption
from task_datasets import ocrDataset, dataset_class_dict
from models import Model_Worker, Web_Model_Worker


def get_model(args):
    device = torch.device('cpu' if args.device == -1 else f"cuda:{args.device}")
    if args.use_web_model:
        model = Web_Model_Worker()
    else:
        model = Model_Worker(device)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--use-web-model",
        action="store_true",
        default=False,
        help="Whether to use web model worker."
    )
    
    # datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP ct80 cocotext ctw totaltext HOST WOST WordArt")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )
    parser.add_argument(
        "--eval_vqa",
        action="store_true",
        default=False,
        help="Whether to evaluate on vqa."
    )
    parser.add_argument(
        "--eval_caption",
        action="store_true",
        default=False,
        help="Whether to evaluate on caption."
    )

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


def main(args):
    model = get_model(args)
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if args.eval_ocr:
        ocr_dataset_name = args.ocr_dataset_name.split()
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(ocr_dataset_name[i])
            acc = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time)
            result[ocr_dataset_name[i]] = acc

    if args.eval_vqa:
        dataset_class = dataset_class_dict[args.dataset_name]
        dataset = dataset_class()
        dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        acc = evaluate_VQA(model, dataset, args.model_name, args.dataset_name, time)
        result[args.dataset_name] = acc
    
    if args.eval_caption:
        dataset_class = dataset_class_dict[args.dataset_name]
        dataset = dataset_class()
        dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        acc = evaluate_Caption(model, dataset, args.model_name, args.dataset_name, time)
        result[args.dataset_name] = acc
    
    result_path = os.path.join(os.path.join(args.answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)