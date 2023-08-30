import json
import numpy as np
import pandas as pd
import os
from utils.tools import VQAEval, has_word, remove_special_chars

available_models = sorted([
    'BLIP2', 'InstructBLIP', 'LLaVA', 'LLaMA-Adapter-v2',
    'MiniGPT-4', 'mPLUG-Owl', 'Otter', 'VPGTrans',
    'Otter-Image', 'PandaGPT', 'OFv2_4BI', 'Bard',
])
num_models = len(available_models)
datasets_interest = sorted(['IC15', 'VSR', 'VCR1_MCI', 'ImageNetVC_shape', 'MSCOCO_pope_adversarial'])

# human evaluation
human = {x: {y: [] for y in available_models} for x in datasets_interest}
for dataset in datasets_interest:
    with open(f"datasets/tiny_lvlm_ehub_random_50samples/human_evaluation/{dataset}_judge_results.jsonl", "r") as f:
        results = [json.loads(x) for x in f.readlines()]
    for x in results:
        models = x['models']
        judges = x['judged']
        for model, judge in zip(models, judges):
            a = int(judge == 'Correct')
            human[dataset][model].append(a)

# gpt evaluation
gpt = {x: {y: [] for y in available_models} for x in datasets_interest}
gpt_root = 'datasets/tiny_lvlm_ehub_random_50samples/gpt_evaluation/tiny_answers_0727_2_ensemble_items_8_10_12_13_14'
for dataset in datasets_interest:
    for model in available_models:
        json_path = os.path.join(gpt_root, model, f'{dataset}.json')
        results = json.load(open(json_path))
        for x in results:
            a = int(x['gpt_score_ensemble'] == 1)
            gpt[dataset][model].append(a)

# hasword evaluation
dataset2task = {
    'IC15': 'OCR',
    'VCR1_MCI': 'VQA',
    'ImageNetVC_shape': 'VQA',
    'VSR': 'VQA',
    'MSCOCO_pope_adversarial': 'VQA',
}
hasword = {x: {y: [] for y in available_models} for x in datasets_interest}
hasword_root = 'datasets/tiny_lvlm_ehub_random_50samples/tiny_answers'
for dataset in datasets_interest:
    task = dataset2task[dataset]
    for model in available_models:
        json_path = os.path.join(hasword_root, model, f'{dataset}.json')
        results = json.load(open(json_path))
        for x in results:
            answer = x['answer']
            gt_answers = x['gt_answers']
            if task == 'OCR':
                gt_answers = remove_special_chars(gt_answers).lower()
                answer = remove_special_chars(answer).lower()
                a = int(has_word(answer, gt_answers))
            elif task == 'VQA':
                vqa = VQAEval()
                a = int(vqa.evaluate(answer, gt_answers))
            hasword[dataset][model].append(a)

# agreement
# collection_a = {x: [] for x in datasets_interest}
# collection_b = {x: [] for x in datasets_interest}
collection = {}
for dataset in datasets_interest:
    collection[f'{dataset}_ChatGPT'] = []
    collection[f'{dataset}_hasword'] = []
total = 0
better = 0
tie = 0
for dataset in datasets_interest:
    for model in available_models:
        gt = np.array(human[dataset][model])

        llm = np.array(gpt[dataset][model])
        a = int(np.mean(gt == llm) * 100)
        collection[f'{dataset}_ChatGPT'].append(a)

        auto = np.array(hasword[dataset][model])
        b = int(np.mean(gt == auto) * 100)
        collection[f'{dataset}_hasword'].append(b)
        if a > b:
            better += 1
        elif a == b:
            tie += 1 
        total +=1
        # collection_a[dataset].append(a)
        # collection_b[dataset].append(b)

print(better, tie, total, better/total)
# df_a = pd.DataFrame(collection_a, index=available_models)
# df_b = pd.DataFrame(collection_b, index=available_models)
# print(df_a)
# print(df_b)
df = pd.DataFrame(collection, index=available_models)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
print(df)


foo = df.iloc[:, 0::2].to_numpy().mean(axis=1)
boo = df.iloc[:, 1::2].to_numpy().mean(axis=1)

models_ = ['BLIP2' , 'InstructBLIP' , 'LLaMA-Adapter-v2' , 'LLaVA' , 'MiniGPT-4' , 'mPLUG-Owl' , 'OFv2_4BI', 'Otter' ,'Otter-Image' , 'PandaGPT' , 'VPGTrans' , 'Bard']
print(models_)
foo = [foo[available_models.index(x)] for x in models_]
boo = [boo[available_models.index(x)] for x in models_]

print(boo)
print(foo)
print([x > y for x, y in zip(foo, boo)])

