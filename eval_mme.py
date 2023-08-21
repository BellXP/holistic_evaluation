import os
import json
import argparse
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import torch
from torch.utils.data import DataLoader

from models import get_model
from task_datasets.mme_benchmark import MMEDataset

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA2-Accessory")
    parser.add_argument("--batch_size", type=int, default=32)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./mme_answers")

    args = parser.parse_args()
    return args


def compute_mme_metric(gts, preds):
    assert len(gts) == len(preds)

    label_map = {
        "yes": 1,
        "no": 0,
        "other": -1,
    }
    
    gts = [label_map[x] for x in gts]
    preds = [label_map[x] for x in preds]

    acc = accuracy_score(gts, preds) 

    clean_gts = []
    clean_preds = []
    other_num = 0 
    for gt, pred in zip(gts, preds):
        if pred == -1:
            other_num += 1
            continue
        clean_gts.append(gt)
        clean_preds.append(pred)
    
    conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
    precision = precision_score(clean_gts, clean_preds, average='binary')
    recall = recall_score(clean_gts, clean_preds, average='binary')
    tp, fn = conf_mat[0]
    fp, tn = conf_mat[1]

    metric_dict = dict()
    metric_dict = {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "precision": precision,
        "recall": recall,
        "other_num": other_num,
        "acc": acc,
    }

    return metric_dict


def evaluate_mme(model, dataset, model_name, dataset_name, batch_size, answer_path):
    os.makedirs(answer_path, exist_ok=True)
    answer_path = os.path.join(answer_path, f"{dataset_name}.json")
    
    if not os.path.exists(answer_path):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            outputs = model.batch_generate(batch['image_path'], batch['question'])
            for image_path, question, gt_answer, output in zip(batch['image_path'], batch['question'], batch['gt_answers'], outputs):
                answer_dict={'question': question, 'answer': output, 'gt_answers': gt_answer, 'image_path': image_path, 'model_name': model_name}
                predictions.append(answer_dict)

        if torch.distributed.get_rank() != 0:
            return
        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))

    gts = []
    preds = []
    img_correct_num = 0
    task_other_ans_num = 0
    acc_plus_correct_num = 0
    last_correct = False
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers'].lower()
            answer = dict[i]['answer'].lower()
            assert gt_answers in ['yes', 'no']
            answer = 'yes' if 'yes' in answer else 'no' if 'no' in answer else 'other'
            gts.append(gt_answers)
            preds.append(answer)

            if gt_answers == answer:
                img_correct_num += 1
                if (i + 1) % 2 == 0 and last_correct:
                    acc_plus_correct_num += 1
                last_correct = True
            else:
                last_correct = False

            if answer == 'other':
                task_other_ans_num += 1

        metric_dict = compute_mme_metric(gts, preds)
        metric_dict['acc_plus'] = (acc_plus_correct_num / 2) / len(dict)

    task_score = 0
    for k, v in metric_dict.items():
        if k in ["acc", "acc_plus"]:
            task_score += v * 100

    print(f"{dataset_name}: {task_score}")
    return task_score


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    dataset_names = ["artwork", "celebrity", "color", "count", "existence", "OCR", "position", "posters", "scene", "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation", "landmark"] # landmark (03d5e3bfc958be38.jpg)
    for dataset_name in dataset_names:
        dataset = MMEDataset(dataset_name)
        metrics = evaluate_mme(model, dataset, args.model_name, dataset_name, args.batch_size, answer_path)
        result[dataset_name] = metrics

    if torch.distributed.get_rank() != 0:
        return
    result_path = os.path.join(answer_path, 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)