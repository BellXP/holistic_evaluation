import os
import json


model_names = [
    'LLaVA', 'Lynx', 'VPGTrans',
    'BLIP2', 'InstructBLIP-T5', 'InstructBLIP', 'BLIVA',
    'LLaMA-Adapter-v2', 'Otter-Image', 'Cheetah'
]

task_dataset_names = {
    'Visual Perception': [
        'ImageNet', 'CIFAR10', 'OxfordIIITPet', 'Flowers102',
        'VCR1_OC', 'VCR1_MCI', 'MSCOCO_OC', 'MSCOCO_MCI'
    ],
    'Visual Knowledge Acquisition': [
        "IIIT5K", "IC13", "IC15", "Total-Text", "CUTE80",
        "SVT", "SVTP", "COCO-Text", "WordArt", "CTW",
        "HOST", "WOST", 'SROIE', 'FUNSD'
    ],
    'Visual Reasoning': [
        'DocVQA', 'TextVQA', 'STVQA', 'OCRVQA', 'OKVQA',
        'GQA', 'IconQA', 'VSR', 'WHOOPS', 'ScienceQA', 'VizWiz'
    ],
    'Visual Commonsense': [
        'ImageNetVC_others', 'ImageNetVC_color', 'ImageNetVC_shape',
        'ImageNetVC_material', 'ImageNetVC_component', 'VCR'
    ],
    'Object Hallucination': [
        'MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial'
    ]
}


def has_word(sentence, word):
    import re
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False


def calcu_pope_metrics(model_outputs):
    label_list, pred_list = [], []
    for model_output in model_outputs:
        gt_answer = model_output['gt_answers']
        if gt_answer == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

        answer = model_output['answer']
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()

        if has_word(answer.lower(), gt_answer.lower()):
            if gt_answer == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)
        else:
            if gt_answer == 'no':
                pred_list.append(1)
            else:
                pred_list.append(0)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc, precision, recall, f1, yes_ratio



def main(task_name):
    # get results dict
    results_root = '/mnt/lustre/xupeng/workplace/holistic_evaluation/answers'
    results_dict = {}
    for model_name in model_names:
        model_path = f"{results_root}/{model_name}"
     
        for subdir in os.listdir(model_path):
            result_file = f"{model_path}/{subdir}/result.json"
            if not os.path.exists(result_file):
                continue
            subresults = json.load(open(result_file))
            for subset in subresults:
                if subset in ['MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial']:
                    model_outputs = f"{model_path}/{subdir}/{subset}.json"
                    model_outputs = json.load(open(model_outputs))
                    acc, precision, recall, f1, yes_ratio = calcu_pope_metrics(model_outputs)
                    subresults[subset] = acc

                if subset not in results_dict:
                    results_dict[subset] = {}
                if model_name in results_dict[subset]:
                    # print(f"Duplicate result in {subset},{model_name}: {results_dict[subset][model_name]} --> {100 * subresults[subset]}")
                    subresults[subset] = max(subresults[subset], results_dict[subset][model_name] / 100)
                results_dict[subset][model_name] = 100 * subresults[subset]

    # # create task result table (Markdown)
    # print(f"### {task_name}")
    # print(f"| Model | {' | '.join(task_dataset_names[task_name])} |")
    # print(f"| :--: | {':--: | ' * len(task_dataset_names[task_name])}")
    # for model_name in model_names:
    #     print(f"| {model_name} | ", end='')
    #     for subset in task_dataset_names[task_name]:
    #         try:
    #             print(f"{results_dict[subset][model_name]:.2f}", end=' | ')
    #         except:
    #             print("XXX", end=' | ')
    #     print()

    # create task result table (LaTEX)
    print(f"### {task_name}")
    print(f"& Datasets & {' & '.join(model_names)} \\\\")
    avg_scores = []
    for subset in task_dataset_names[task_name]:
        print(f"& {subset} & ", end='')
        row_scores = []
        for model_name in model_names:
            try:
                subset_score = results_dict[subset][model_name]
                print(f"{subset_score:.2f}", end=' & ')
                row_scores.append(subset_score)
            except:
                print("XXX", end=' & ')
        print()
        row_scores = [x / max(row_scores) for x in row_scores]
        avg_scores.append(row_scores)
    print(f"& Average Score & ", end='')
    for i, model_name in enumerate(model_names):
        model_scores = [x[i] for x in avg_scores]
        model_avg_score = sum(model_scores) / len(model_scores)
        print(f"{100 * model_avg_score:.2f}", end=' & ')
    print()


if __name__ == "__main__":
    # main('Visual Perception')
    # main('Visual Knowledge Acquisition')
    # main('Visual Reasoning')
    # main('Visual Commonsense')
    main('Object Hallucination')
