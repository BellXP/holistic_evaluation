import json
from pathlib import Path
import os
import pandas as pd
import re
from collections import defaultdict

from utils.tools import VQAEval, has_word, remove_special_chars
from utils.kie import F1Scorer
from utils.cider import CiderScorer

root_dir = Path('datasets/tiny_lvlm_ehub_random_50samples')
bard_evaluation = root_dir / 'bard_evaluation'

jsonls = sorted(list(bard_evaluation.glob('*.jsonl')))

vqa = VQAEval()

remove_mesg = ["I can't process this file", "Sorry, I can't help with images of people yet"]

target_num = 50

tasks = []
datasets = []
accs = []
all_captions = {}
for x in jsonls:
    y = x.name.split('_')
    task = y[0]
    dataset = '_'.join(y[1:-2])
    if dataset != 'COCO-Text':
        continue
    if dataset in ['ScienceQA', 'WHOOPSWeird']:
        continue
    # tasks.append(task)
    # datasets.append(dataset)
    with open(x, 'r') as f:
        infers = [json.loads(x) for x in f.readlines()]
    if task not in ['OCR', 'KIE', 'Caption', 'Demo', 'Bug', 'MCI'] and dataset not in ['HatefulMemes', 'VSR', 'RSVQALR_mci', 'COD10K']:
        tasks.append(task)
        datasets.append(dataset)
        num_correct = 0
        num_infer = 0
        num_skip = 0
        per_class = False
        if dataset in ['Flowers102', 'OxfordIIITPet']:
            per_class_dict = defaultdict(lambda : defaultdict(int))
            per_class = True

        for a in infers:
            if num_infer == target_num:
                break
            candidate = a['bard']
            if any([b in candidate for b in remove_mesg]):
                num_skip += 1
                continue
            gt_answers = a['gt_answers']
            if isinstance(gt_answers, list):
                gt_answers = [b.rstrip('.') for b in gt_answers]
            else:
                gt_answers = gt_answers.rstrip('.')
            if per_class:
                per_class_dict[gt_answers]['total'] += 1
            if vqa.evaluate(candidate, gt_answers):
                num_correct += 1
                if per_class:
                    per_class_dict[gt_answers]['correct'] += 1
            num_infer += 1
            print('>>>', candidate)
            print('>>>', gt_answers)
        assert num_infer == target_num, f'{dataset}'
        accuracy = num_correct / num_infer * 100
        if per_class:
            num_classes = len(per_class_dict)
            acc_sum = 0.0
            for val in per_class_dict.values():
                acc_sum += val['correct'] / val['total']
            accuracy = acc_sum / num_classes * 100
        accs.append(round(accuracy, 2))
    elif dataset in ['HatefulMemes', 'VSR', 'MSCOCO_MCI', 'VCR1_MCI', 'RSVQALR_mci', 'COD10K']:
        tasks.append(task)
        datasets.append(dataset)
        num_correct = 0
        num_skip = 0
        num_infer = 0

        for a in infers:
            if num_infer == target_num:
                break
            candidate = a['bard']
            if any([b in candidate for b in remove_mesg]):
                num_skip += 1
                continue
            if 'Answer:' not in candidate:
                num_skip += 1
                continue
            pattern = r"Answer:\*{0,2} \*{0,2}(Yes|No)"
            try:
                candidate = re.findall(pattern, candidate, re.DOTALL)[0]
            except:
                assert False, candidate
            gt_answers = a['gt_answers'].rstrip('.')
            if candidate.lower() == gt_answers.lower():
                num_correct += 1
            num_infer += 1
        assert num_infer == target_num, f'{dataset}'

        # assert (len(infers)-num_skip) == 50, f'less than 50 samples'
        accuracy = num_correct / (len(infers)-num_skip) * 100
        accs.append(round(accuracy, 2))
    elif task == 'OCR':
        tasks.append(task)
        datasets.append(dataset)
        num_correct = 0
        num_skip = 0
        num_infer = 0
        for a in infers:
            if num_infer == target_num:
                break
            candidate = a['bard']
            if any([b in candidate for b in remove_mesg]):
                num_skip += 1
                continue
            gt_answers = remove_special_chars(a['gt_answers']).lower()
            candidate = remove_special_chars(candidate).lower()
            if has_word(candidate, gt_answers):
                num_correct += 1
            num_infer += 1
        assert num_infer == target_num, f'{dataset}'
        accuracy = num_correct / num_infer * 100
        accs.append(round(accuracy, 2))
    elif task == 'KIE':
        if dataset == 'POIE':
            continue
        tasks.append(task)
        datasets.append(dataset)
        # f1_scorer = F1Scorer()
        num_correct = 0
        num_skip = 0
        num_infer = 0
        for a in infers:
            if num_infer == target_num:
                break
            candidate = a['bard']
            gt_answers = a['gt_answers']
            if any([b in candidate for b in remove_mesg]):
                num_skip += 1
                continue
            if has_word(candidate.lower(), gt_answers.lower()):
                num_correct += 1
            num_infer += 1
        assert num_infer == target_num, f'{dataset}'
            # f1_scorer.add_string(gt_answers, candidate)
        # print(f1_scorer.n_match_words)
        # prec, recall, f1 = f1_scorer.score()
        accuracy = num_correct / num_infer * 100
        # accs.append(round(f1, 2))
        accs.append(round(accuracy, 2))
    elif task == 'Caption':
        tasks.append(task)
        datasets.append(dataset)
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        num_infer = 0
        for a in infers:
            if num_infer == target_num:
                break
            candidate = a['bard']
            gt_answers = a['gt_answers']
            if any([b in candidate for b in remove_mesg]):
                num_skip += 1
                continue
            pattern = r"\r?\n\r?\n\*{0,2}>?([a-zA-Z0-9_\- ,\"\'\(\)]+)\."
            try:
                candidate = re.findall(pattern, candidate, re.DOTALL)[0]
            except:
                # print('>>>', candidate)
                pass
            if candidate.startswith('The image shows '):
                candidate = candidate[len('The image shows '):]
            candidate = candidate.strip().capitalize()
            if candidate in all_captions:
                print(task, dataset, candidate)
            else:
                all_captions[candidate] = True
            gt_answers = [a.rstrip('.').strip().capitalize() for a in gt_answers]
            cider_scorer += (candidate, gt_answers)
            num_infer += 1
        assert num_infer == target_num, f'{dataset}'
        score, scores = cider_scorer.compute_score()
        accs.append(round(score * 100, 2))
        # print(f'Task: {task}, Dataset: {dataset}, accuracy: {accuracy:.2f}%')
df = pd.DataFrame({'Task': tasks, 'Dataset': datasets, 'Accuracy': accs})
print(df)
