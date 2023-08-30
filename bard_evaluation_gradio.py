import json
from pathlib import Path
import gradio as gr

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

remove_mesg = ["I can't process this file", "Sorry, I can't help with images of people yet"]

root_dir = Path('datasets/tiny_lvlm_ehub_random_50samples')
jsonl_path = root_dir / 'samples_jsonl'
bard_evaluation = root_dir / 'bard_evaluation'
dataset_jsonls = sorted(list(jsonl_path.glob('*_400samples.jsonl')))
target_num = 50
all_datasets = {}
for x in dataset_jsonls:
    y = x.name.split('_')
    task = y[0]
    dataset = '_'.join(y[1:-2])
    if task not in all_datasets:
        all_datasets[task] = {}
    all_datasets[task][dataset] = str(x)

with open('datasets/tiny_lvlm_ehub_random_50samples/bard_RandomSeed20230719_50samples_per_dataset.jsonl', 'r') as f:
    dataset_indices = [json.loads(x) for x in f.readlines()]
    dataset_indices = {x['dataset']: x['indices'] for x in dataset_indices}

def task2dataset(state, task):
    datasets = sorted(list(all_datasets[task].keys()))
    state['task'] = task
    return state, gr.Dropdown.update(value='', choices=datasets, interactive=True)

def dataset2start(state, dataset):
    task = state['task']
    state['dataset'] = dataset
    samples = []
    samples_jsonl = all_datasets[task][dataset]
    state['samples_jsonl'] = samples_jsonl
    with open(samples_jsonl, 'r') as f:
        samples = [json.loads(x) for x in f.readlines()]
    state['samples'] = samples
    metadata_json = bard_evaluation / f'{task}_{dataset}_metadata.json'
    if metadata_json.exists():
        metadata = json.load(open(metadata_json))
    else:
        metadata = {}
        metadata['task'] = task
        metadata['dataset'] = dataset
        metadata['num_completed'] = 0
        metadata['num_samples'] = len(samples)
        metadata['evaluation_jsonl'] = str(bard_evaluation / f'{task}_{dataset}_evaluation_results.jsonl')
        json.dump(metadata, open(metadata_json, 'w'))
    metadata['num_samples'] = len(samples)
    state['metadata_json'] = metadata_json
    state['metadata'] = metadata
    state['current_sample_id'] = metadata['num_completed']
    if state['dataset'] == 'HatefulMemes':
        state['current_sample_id'] = dataset_indices['HatefulMemes'][-1] + 1
    state['num_completed_valid'] = 0
    evaluation_jsonl = root_dir / metadata['evaluation_jsonl']
    # print(evaluation_jsonl)
    if evaluation_jsonl.exists():
        with open(evaluation_jsonl, 'r') as f:
            infers = [json.loads(a) for a in f.readlines()]
        num_removed = 0
        for a in infers:
            if any([b in a['bard'] for b in remove_mesg]):
                num_removed += 1
                continue
            if metadata['dataset'] == 'HatefulMemes':
                if 'Answer:' not in a['bard']:
                    num_removed += 1
                    continue
        state['num_completed_valid'] = len(infers) - num_removed
        state['current_sample_id'] += len(infers) - 50
    return state, gr.update(interactive=False), enable_btn

def answer_submit(state, answer):
    state['bard'] = answer
    return state, enable_btn

def start_anno(state):
    idx = state['current_sample_id']

    if state['Start']:
        idx = state['current_sample_id']
        state['metadata']['num_completed'] = idx
        metadata_json = state['metadata_json']
        json.dump(state['metadata'], open(metadata_json, 'w'))

        sample = state['samples'][idx-1]
        sample['bard'] = state['bard']
        evaluation_jsonl = root_dir / state['metadata']['evaluation_jsonl']
        with open(evaluation_jsonl, 'a') as f:
            f.write(json.dumps(sample) + '\n')
        if all([b not in sample['bard'] for b in remove_mesg]):
            if state['metadata']['dataset'] == 'HatefulMemes':
                if 'Answer:' in sample['bard']:
                    state['num_completed_valid'] += 1
            else:
                state['num_completed_valid'] += 1

    num_samples = state['metadata']['num_samples']
    done = state['num_completed_valid']
    left = target_num - done
    completion = [(done, 'done'), (left, 'left')]

    if idx == num_samples or left <= 0:
        next_btn = gr.Button.update(value='Well done 🥳🥳🥳', interactive=False)
        text = gr.Textbox.update(value='', interactive=False)
        return None, text, text, next_btn,completion, gr.update(interactive=False)

    sample = state['samples'][idx]
    state['current_sample_id'] = idx + 1
    image_path = str(root_dir / sample['image_path'])
    question = sample['question']
    answer = gr.Textbox.update('', interactive=True)
    state['question'] = question
    state['image_path'] = image_path
    next_btn = gr.Button.update(value='Next', interactive=False)
    state['Start'] = True
    dataset = gr.update(interactive=False)
    return image_path, question, answer, next_btn, completion, dataset
remaining = {
    # 'Caption': {'Flickr': 26, 'person': 'shaowenqi'},
    # 'MCI': {'VCR1_MCI': 16, 'person': 'shaowenqi'},
    # 'OC': {'VCR1_OC': 13, 'person': 'leimeng'},
    # 'POPE': {'MSCOCO_pope_adversarial': 5, 'MSCOCO_pope_popular': 5, 'person': 'leimeng'},
    'VQA': {'AOKVQAClose': 4, 'person': 'leimeng',
    'AOKVQAOpen': 14, 'person1': 'mengfanqing',
    'DocVQA': 2, 'person2': 'mengfanqing',
    'GQA': 21, 'person3': 'mengfanqing',
    'HatefulMemes': 35, 'person4': 'huyutao',
    'OCRVQA': 13, 'person5': 'huyutao',
    'OKVQA': 13, 'person6': 'xupeng',
    'STVQA': 11, 'person7': 'xupeng',
    'ScienceQAIMG': 1, 'person8': 'xupeng',
    'TextVQA': 10, 'person9': 'xupeng',
    'VQAv2': 13, 'person10': 'xupeng',
    'VSR': 11, 'person11': 'zhangkaipeng',
    'Visdial': 18, 'person12': 'zhangkaipeng',
    'VizWiz': 4, 'person13': 'xupeng',
    'WHOOPSVQA': 21, 'person14': 'zhangkaipeng',
    'WHOOPSWeird': 19, 'person15': 'shaowenqi',
    }}

# foo = {'Caption': {'Flickr': 26},
#  'MCI': {'VCR1_MCI': 16},
#  'OC': {'VCR1_OC': 13},
#  'POPE': {'MSCOCO_pope_adversarial': 5, 'MSCOCO_pope_popular': 5},
#  'VQA': {'AOKVQAClose': 4,
#   'AOKVQAOpen': 14,
#   'DocVQA': 2,
#   'GQA': 21,
#   'HatefulMemes': 35,
#   'OCRVQA': 13,
#   'OKVQA': 13,
#   'STVQA': 11,
#   'ScienceQAIMG': 1,
#   'TextVQA': 10,
#   'VQAv2': 13,
#   'VSR': 11,
#   'Visdial': 18,
#   'VizWiz': 4,
#   'WHOOPSVQA': 21,
#   'WHOOPSWeird': 19}}

with gr.Blocks() as demo:
    state = gr.State({'Start': False})
    with gr.Row():
        with gr.Column(scale=1):
            tasks = sorted(list(all_datasets.keys()))
            task = gr.Dropdown(tasks, label="Task")
        with gr.Column(scale=1):
            dataset = gr.Dropdown([], label="Dataset", interactive=False)
    with gr.Row():
        with gr.Column(scale=1):
            next_btn = gr.Button('Start', interactive=False)
            with gr.Accordion("🔔 README.md", open=False):
                gr.JSON(remaining)
        with gr.Column(scale=1):
            completion = gr.HighlightedText(label="Completion",
            ).style(color_map={"done": "green", "left": "red"})
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(interactive=False)
        with gr.Column(scale=1):
            question = gr.Textbox(label='Question')
            answer = gr.Textbox(label='Answer')
    task.select(task2dataset, [state, task], [state, dataset])
    dataset.select(dataset2start, [state, dataset], [state, task, next_btn])
    next_btn.click(start_anno, state, [image, question, answer, next_btn, completion, dataset])
    answer.submit(answer_submit, [state, answer], [state, next_btn])

demo.launch(share=True)
