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
    num = 0
    no_gts = 0
    wrong_yes, correct_yes = 0, 0
    wrong_no, correct_no = 0, 0
    for i in range(len(predictions)):
        gt_answers = predictions[i]['gt_answers']
        answer = predictions[i]['answer']
        has_yes = eval.evaluate(answer, 'yes') == 1
        has_no = eval.evaluate(answer, 'no') == 1
        
        if has_yes:
            if gt_answers == 'yes':
                correct_yes += 1
            else:
                wrong_yes += 1
        if has_no:
            if gt_answers == 'no':
                correct_no += 1
            else:
                wrong_no += 1
        if not has_yes and not has_no:
            no_gts += 1
        num+=1

    num_no = correct_no + wrong_no
    num_yes = correct_yes + wrong_yes
    correct = correct_yes + correct_no
    print(f'Total num: {num}; Correct: {correct}; No any gts: {no_gts}')
    print(f'Accuracy: {float(correct)/num}')
    print(f'Ratio of no gts: {float(no_gts)/num}')
    print(f'Ratio of yes: {float(num_yes)/num}')
    print(f'Ratio of no: {float(num_no)/num}')
    print(f"Ratio of correct yes: {float(correct_yes)/num}")
    print(f"Ratio of correct no: {float(correct_no)/num}")
    print(f"Ratio of wrong yes: {float(wrong_yes)/num}")
    print(f"Ratio of wrong no: {float(wrong_no)/num}")


for suffix in ['', '_Yes', '_No']:
    for dataset_name in ['MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial']:
        data_path = f"answers/mPLUG-Owl-POPE/{dataset_name}{suffix}.json"
        print(f"{dataset_name}{suffix}")
        eval_yes_no(data_path)
        print()
