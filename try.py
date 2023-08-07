import json
from utils.tools import VQAEval


# answer_path = 'answers/G2PT-7B/20230708152542'
# predictions = []
# for i in range(8):
#     data_path = f"{answer_path}/OCR{i}.json"
#     with open(data_path, 'r') as f:
#         data = json.load(f)
#     predictions.extend(data)
#     print(len(predictions))

# with open(f"{answer_path}/OCR.json", "w") as f:
#     f.write(json.dumps(predictions, indent=4))


def eval_yes_no(data_path):
    with open(data_path, 'r') as f:
        predictions = json.load(f)

    eval = VQAEval()
    correct = 0
    no_gts = 0
    num_yes = 0
    num_no = 0
    num = 0
    for i in range(len(predictions)):
        gt_answers = predictions[i]['gt_answers']
        answer = predictions[i]['answer']
        if eval.evaluate(answer, gt_answers) == 1:
            correct += 1
        has_yes = eval.evaluate(answer, 'yes') == 1
        has_no = eval.evaluate(answer, 'no') == 1
        if has_yes:
            num_yes += 1
        if has_no:
            num_no += 1
        if not has_yes and not has_no:
            no_gts += 1

        num+=1

    print(f'Total num: {num}; Correct: {correct}; No any gts: {no_gts}')
    print(f'Accuracy: {float(correct)/num}')
    print(f'Ratio of no gts: {float(no_gts)/num}')
    print(f'Ratio of yes: {float(num_yes)/num}')
    print(f'Ratio of no: {float(num_no)/num}')


for suffix in ['', '_Yes', '_No']:
    for dataset_name in ['MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial']:
        data_path = f"answers/mPLUG-Owl-POPE/{dataset_name}{suffix}.json"
        print(f"{dataset_name}{suffix}")
        eval_yes_no(data_path)
        print()
