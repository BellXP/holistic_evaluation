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


data_path = 'tiny_answers/Otter/20230726104105/HatefulMemes.json'
with open(data_path, 'r') as f:
    predictions = json.load(f)


eval = VQAEval()
correct = 0
num = 0
for i in range(len(predictions)):
    gt_answers = predictions[i]['gt_answers']
    answer = predictions[i]['answer']
    if eval.evaluate(answer, gt_answers)==1:
        correct+=1
    num+=1
print(f'{float(correct)/num}')
