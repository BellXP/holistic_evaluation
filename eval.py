import os
import json
import argparse
import datetime

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA
from datasets.ocr_dataset import ocrDataset
from datasets.vqa_dataset import textVQADataset, docVQADataset, ocrVQADataset, STVQADataset

from model_worker import Model_Worker
from web_model_worker import Web_Model_Worker


def get_model(args):
    device = args.device
    model = Web_Model_Worker()

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    #OCR datasets
    parser.add_argument("--ocr_dir_path", type=str, default="/home/xupeng/workplace/OCR_Datasets")
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP ct80 cocotext ctw totaltext HOST WOST WordArt")
    #textVQA
    parser.add_argument("--textVQA_image_dir_path", type=str, default="./data/textVQA/train_images")
    parser.add_argument("--textVQA_ann_path", type=str, default="./data/textVQA/TextVQA_0.5.1_val.json")

    #docVQA
    parser.add_argument("--docVQA_image_dir_path", type=str, default="./data/docVQA/val")
    parser.add_argument("--docVQA_ann_path", type=str, default="./data/docVQA/val/val_v1.0.json")

    #ocrVQA
    parser.add_argument("--ocrVQA_image_dir_path", type=str, default="./data/ocrVQA/images")
    parser.add_argument("--ocrVQA_ann_path", type=str, default="./data/ocrVQA/dataset.json")

    #STVQA
    parser.add_argument("--STVQA_image_dir_path", type=str, default="./data/STVQA")
    parser.add_argument("--STVQA_ann_path", type=str, default="./data/STVQA/train_task_3.json")

    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )

    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()
    return args


def main(args):
    np.random.seed(0)
    max_sample_num = 5000

    model = get_model(args)

    '''ocr_dataset_name=['IIIT5K','svt','IC13_857','IC15_1811','svtp','ct80',
                  'cocotext','ctw','totaltext','HOST','WOST','WordArt']'''
    ocr_dataset_name = args.ocr_dataset_name.split()
    
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if args.eval_textVQA:
        dataset = textVQADataset(args.textVQA_image_dir_path, args.textVQA_ann_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time)
        result['textVQA'] = acc
    if args.eval_docVQA:
        dataset = docVQADataset(args.docVQA_image_dir_path, args.docVQA_ann_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time)
        result['docVQA'] = acc

    if args.eval_ocrVQA:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset,random_indices)
        acc = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time)
        result['ocrVQA'] = acc
    
    if args.eval_STVQA:
        dataset = STVQADataset(args.STVQA_image_dir_path, args.STVQA_ann_path)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset,random_indices)
        acc = evaluate_VQA(model, dataset, args.model_name, 'STVQA', time)
        result['STVQA'] = acc

    if args.eval_ocr:
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(args.ocr_dir_path, ocr_dataset_name[i])
            acc = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time)
            result[ocr_dataset_name[i]] = acc
    
    result_path = os.path.join(os.path.join(args.answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)