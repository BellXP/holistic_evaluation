import os
import json
import argparse
import datetime

import torch
import numpy as np

from models import get_model
from utils import dataset_task_dict
from utils.kie import F1Scorer
from utils.caption import CiderScorer
from utils.tools import VQAEval
from tiny_datasets import dataset_class_dict, GeneralDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")

    args = parser.parse_args()
    return args


def eval_kie(file_name, task_type):
    with open(file_name, 'r') as f:
        dict = json.load(f)
        f1_scorer = F1Scorer()
        for i in range(len(dict)):
            dict[i]['task_type'] = task_type
            answer = dict[i]['answer']
            gt_answers = dict[i]['gt_answers']
            f1_scorer.add_string(gt_answers, answer)
        prec, recall, f1 = f1_scorer.score()
    result = f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"

    with open(file_name, "w") as f:
        f.write(json.dumps(dict, indent=4))
    return result


def eval_caption(file_name, task_type):
    with open(file_name, 'r') as f:
        dict = json.load(f)
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        for i in range(len(dict)):
            dict[i]['task_type'] = task_type
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            cider_scorer += (answer, gt_answers)
        (score, scores) = cider_scorer.compute_score()
    for i, sample_score in zip(range(len(dict)), scores):
        dict[i]['cider_score'] = sample_score

    with open(file_name, "w") as f:
        f.write(json.dumps(dict, indent=4))
    
    return score


def eval_vqa(file_name, task_type):
    eval = VQAEval()
    correct = 0
    num = 0
    with open(file_name, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            dict[i]['task_type'] = task_type
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer, gt_answers)==1:
                correct+=1
            num+=1

    with open(file_name, "w") as f:
        f.write(json.dumps(dict, indent=4))
    
    return float(correct)/num


def main(args):
    result = {}
    dataset_names = args.dataset_name.split(',')
    for dataset_name in dataset_names:
        _, task_type = dataset_task_dict[dataset_name]
        file_name = f"{args.answer_path}/{args.model_name}/{dataset_name}.json"
        if task_type == 'Caption':
            metric = eval_caption(file_name, task_type)
        else:
            metric = eval_vqa(file_name, task_type)
        result[dataset_name] = metric

    result_path = os.path.join(os.path.join(args.answer_path, args.model_name), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)