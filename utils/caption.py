import more_itertools
from tqdm import tqdm
import os
import json

from .cider import CiderScorer


def evaluate_Caption(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='what is described in the image?',
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            cider_scorer += (answer, gt_answers)
        (score, scores) = cider_scorer.compute_score()
    for i, sample_score in zip(range(len(dict)), scores):
        dict[i]['cider_score'] = sample_score
    with open(answer_path, "w") as f:
        f.write(json.dumps(dict, indent=4))
    
    print(f'{dataset_name}: {score}')
    return score