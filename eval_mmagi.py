import os
import io
import json
import base64
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from models import get_model


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


class MMAGIBenchDataset(Dataset):
    def __init__(self,
                 sys_prompt='There are several options:',
                 split: int = -1
        ):
        data_file=f"mmagi_v030_{'test' if USE_TEST_SET else 'dev'}_inferin.tsv"
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        
        assert split <= (MAX_SPLIT - 1), f"Invalid split index: {split}"
        if split < 0:
            self.data_num = len(self.df)
            self.prev_num = 0
        else:
            # split dataset into four parts
            part_num = int(len(self.df) / MAX_SPLIT)
            self.prev_num = part_num * split
            self.data_num = len(self.df) - self.prev_num if split == (MAX_SPLIT - 1) else part_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        idx += self.prev_num
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = None if USE_TEST_SET else self.df.iloc[idx]['answer']
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
            'input_text': f"<image> {question} There are several options: {options_prompt}"
        }
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


def eval_mmagi(
    model,
    dataset,
    batch_size=1,
    answer_path='./answers',
    max_new_tokens=1024,
    split=-1
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['img'], batch['input_text'], max_new_tokens=max_new_tokens, temperature=0)
        for i in range(len(outputs)):
            im_file = io.BytesIO()
            batch['img'][i].save(im_file, format="JPEG")
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes)
            data = {
                'img': im_b64.decode('utf-8'), # encode utf-8 before getting image
                'question': batch['question'][i],
                'answer': batch['answer'][i],
                'options': batch['options'][i],
                'category': batch['category'][i],
                'l2-category': batch['l2-category'][i],
                'options_dict': batch['options_dict'][i],
                'index': str(batch['index'][i]),
                'context': batch['context'][i],
                'input_text': batch['input_text'][i],
                'prediction': outputs[i]
            }
            predictions.append(data)
    os.makedirs(answer_path, exist_ok=True)
    answer_path = os.path.join(answer_path, f"mmagi_{'test' if USE_TEST_SET else 'dev'}_{args.model_name}{'' if split == -1 else f'_{split}'}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", type=str, default="ImageBind")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--max-split", type=int, default=4)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--merge", action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    dataset = MMAGIBenchDataset(split=args.split)
    eval_mmagi(model, dataset, args.batch_size, split=args.split)


def merge_files():
    answer_path = 'answers'
    file_name = f"{answer_path}/mmagi_{'test' if USE_TEST_SET else 'dev'}_{args.model_name}"
    predictions = []
    if os.path.exists(f"{file_name}.json"):
        with open(f"{file_name}.json", 'r') as f:
            predictions = json.load(f)
            print(len(predictions))
    else:
        for i in range(MAX_SPLIT):
            data_path = f"{file_name}_{i}.json"
            with open(data_path, 'r') as f:
                data = json.load(f)
            predictions.extend(data)
            print(len(predictions))

    with open(f"{file_name}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))
    
    json2xlsx(file_name, predictions)


def json2xlsx(file_name, predictions):
    # import pdb; pdb.set_trace()
    for i in range(len(predictions)):
        # predictions[i]['prediction'] = predictions[i]['model_output']
        # del predictions[i]['model_output']
        # del predictions[i]['img']
        options_dict = predictions[i]['options_dict']
        for key in options_dict:
            predictions[i][key] = options_dict[key]
    predictions = pd.DataFrame(predictions)
    predictions.to_excel(f"{file_name}.xlsx", index=None)


if __name__ == "__main__":
    args = parse_args()
    USE_TEST_SET = args.test
    MAX_SPLIT = args.max_split
    if args.merge:
        merge_files()
    else:
        main(args)