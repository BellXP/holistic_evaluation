import os
import json
import argparse
import datetime

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption
from task_datasets import ocrDataset, textVQADataset, STVQADataset, ScienceQADataset, ocrVQADataset, NoCapsDataset
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
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--use-web-model",
        action="store_true",
        default=False,
        help="Whether to use web model worker."
    )
    
    #OCR datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP ct80 cocotext ctw totaltext HOST WOST WordArt")

    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )
    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )
    parser.add_argument(
        "--eval_ScienceQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ScienceQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_NoCaps",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    np.random.seed(seed)
    if len(dataset) > max_sample_num:
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

    if args.eval_textVQA:
        dataset = textVQADataset()
        dataset = sample_dataset(dataset)
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time)
        result['textVQA'] = acc

    if args.eval_STVQA:
        dataset = STVQADataset()
        dataset = sample_dataset(dataset, 4000)
        acc = evaluate_VQA(model, dataset, args.model_name, 'STVQA_4000', time)
        result['STVQA_4000'] = acc

    if args.eval_ScienceQA:
        dataset = ScienceQADataset()
        dataset = sample_dataset(dataset)
        acc = evaluate_VQA(model, dataset, args.model_name, 'ScienceQA', time)
        result['ScienceQA'] = acc

    if args.eval_ocrVQA:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        dataset = sample_dataset(dataset)
        acc = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time)
        result['ocrVQA'] = acc
    
    if args.eval_NoCaps:
        dataset = NoCapsDataset()
        dataset = sample_dataset(dataset)
        acc = evaluate_Caption(model, dataset, args.model_name, 'NoCaps', time)
        result['NoCaps'] = acc
    
    result_path = os.path.join(os.path.join(args.answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)