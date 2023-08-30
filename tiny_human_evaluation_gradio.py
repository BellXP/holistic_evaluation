import gradio as gr
from functools import partial
import random
import json
from pathlib import Path

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

available_models = sorted([
    'BLIP2', 'InstructBLIP', 'LLaVA', 'LLaMA-Adapter-v2',
    'MiniGPT-4', 'mPLUG-Owl', 'Otter', 'VPGTrans',
    'Otter-Image', 'PandaGPT', 'OFv2_4BI', 'Bard',
])
num_models = len(available_models)
# datasets_interest = sorted(['IC15', 'VSR', 'VCR1_MCI', 'ImageNetVC_shape', 'MSCOCO_pope_adversarial'])
datasets_interest = sorted(['IC15', 'VSR', 'VCR1_MCI', 'MSCOCO_pope_adversarial', 'AOKVQAClose'])

root = Path('datasets/tiny_lvlm_ehub_random_50samples')
json_root = root / 'tiny_answers'
image_root = root / 'tiny_lvlm_datasets'
dump_root = root / 'human_evaluation'

# metadata
# dataset: str, metadata_json: str, num_completed: int, judge_jsonl: str

def select_dataset(state, dataset):
    metadata_path = dump_root / f'{dataset}_metadata.json'
    # if dataset == '-1':
    #     Path(metadata_path).unlink()
    if metadata_path.exists():
        metadata = json.load(open(metadata_path))
    else:
        metadata = {}
        metadata['dataset'] = dataset
        metadata['available_models'] = state['available_models']
        metadata['metadata_json'] = str(metadata_path)
        metadata['judge_jsonl'] = str(dump_root / f'{dataset}_judge_results.jsonl')
        metadata['num_completed'] = 0
    state['metadata'] = metadata
    for model in metadata['available_models']:
        samples = json.load(open(json_root / model / f'{dataset}.json'))
        state['samples'][model] = samples
    metadata['num_samples'] = len(state['samples']['Bard'])
    json.dump(metadata, open(metadata_path, 'w'), indent=4)
    if metadata['num_completed'] == metadata['num_samples']:
        start_btn = gr.Button.update(value='Finished 🥳🥳🥳', interactive=False)
        return {}, start_btn
    else:
        return state, gr.Button.update(value='Start', interactive=True)

def save_judge_data(state, judge_jsonl):
    with open(judge_jsonl, 'a') as f:
        f.write(json.dumps(state) + '\n')

def save_metadata(metadata):
    metadata_path = metadata['metadata_json']
    json.dump(metadata, open(metadata_path, 'w'), indent=4)

def start_anno(state):
    metadata = state['metadata']

    if state['start']:
        metadata = state.pop('metadata')
        samples = state.pop('samples')
        save_judge_data(state, metadata['judge_jsonl'])
        metadata['num_completed'] += 1
        save_metadata(metadata)
        state['metadata'] = metadata
        state['samples'] = samples

    if metadata['num_completed'] == metadata['num_samples']:
        models = [''] * num_models
        judges = [gr.Radio.update(value=None, interactive=False)] * num_models
        start_btn = gr.Button.update(value='Finished 🥳🥳🥳', interactive=False)
        progress = {'progress': 0.0}
        done = metadata['num_samples']
        left = 0
        completion = [(done, 'done'), (left, 'left')]
        return {}, start_btn, disable_btn, progress, completion, \
            '', '', None, *models, *judges

    num_completed = metadata['num_completed']
    candidates = []
    models = metadata['available_models'][:]
    random.shuffle(models)
    for model in models:
        total = len(state['samples'][model])
        sample = state['samples'][model][num_completed]
        pred = sample['answer']
        candidates.append(pred)
    # Bard as reference
    sample = state['samples']['Bard'][num_completed]
    question = sample['question']
    reference = sample['gt_answers']
    image_path = sample['image_path']

    state['question'] = question
    state['reference'] = reference
    image_path = str(root / image_path)
    state['image_path'] = image_path
    state['models'] = models
    state['candidates'] = candidates
    state['judged'] = [None] * len(models)
    judges = [gr.Radio.update(value=None, interactive=True)] * num_models
    start_btn = gr.Button.update(value='Next', interactive=False)
    progress = {'progress': 0.0}
    done = num_completed
    left = total - num_completed
    completion = [(done, 'done'), (left, 'left')]
    state['start'] = True
    return state, start_btn, disable_btn, progress, completion, \
        question, reference, image_path, *candidates, *judges

def judge_prediction(state, judge, idx: int):
    num_judged = 0
    if 'judged' in state and judge != None:
        state['judged'][idx] = judge
        num_judged = sum([x is not None for x in state['judged']])

    return state, \
        enable_btn if num_judged == num_models else disable_btn, \
        {'progress': num_judged / num_models}

readme = '''Select dataset "AOKVQAClose" from the dropdown box for demo ONLY.

Available judge options:
- `Correct`: the prediction has the same semantic meaning as the ground-truth answer.
- `Wrong`: the prediction is simply wrong, regardless of hallucinations.
'''

with gr.Blocks(theme=gr.themes.Default()) as demo:
    state = gr.State({'start': False, 'available_models': available_models, 'samples': {}})
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🔔 README.md", open=False):
                gr.Markdown(readme)
            with gr.Row():
                dataset = gr.Dropdown(
                    datasets_interest,
                    label="Dataset",
                    # info="Choose your dataset and then press Start.",
                )
                completion = gr.HighlightedText(label="Completion"
                ).style(color_map={"done": "green", "left": "red"})
            start_btn = gr.Button('Start', interactive=False)
            question = gr.Textbox(label='Question', interactive=False)
            reference = gr.Textbox(label='Answer (ground truth)', interactive=False)
            image = gr.Image()
        with gr.Column(scale=1):
            progress = gr.Label(value={'progress': 0.0}, show_label=False, num_top_classes=1,)
            models = []
            judges = []
            for idx in range(1, num_models+1):
                model = gr.Textbox(label=f'Model#{idx}', show_label=True)
                judge = gr.Radio(["Correct", "Wrong"], show_label=False, interactive=False)
                models.append(model)
                judges.append(judge)
    dataset.select(select_dataset, [state, dataset], [state, start_btn])
    start_btn.click(
        start_anno, state,
        [state, start_btn, dataset, progress, completion, question, reference, image] + models + judges)
    for i, judge in enumerate(judges):
        judge.change(partial(judge_prediction, idx=i), [state, judge], [state, start_btn, progress])

# demo.queue(concurrency_count=16, api_open=False).launch(server_port=7866)
demo.launch(share=True)
