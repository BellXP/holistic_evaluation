import os
import glob
from torch.utils.data import Dataset

DATA_DIR = "/mnt/lustre/xupeng/workplace/MME-Benchmark/MME_Benchmark_release_version"


class MMEDataset(Dataset):
    def __init__(
        self,
        dataset_name
    ):
        self.dataset_name = dataset_name
        self.dataset = []
        jpg_sets = ["artwork", "celebrity", "color", "count", "existence", "landmark", "OCR", "position", "posters", "scene"]
        png_sets = ["code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]
        image_suffix = '.jpg' if dataset_name in jpg_sets else ".png"

        assert (dataset_name in jpg_sets) or (dataset_name in png_sets), f"Invalid dataset name for MME benchmark: {dataset_name}"

        if os.path.exists(f"{DATA_DIR}/{dataset_name}/images") and os.path.exists(f"{DATA_DIR}/{dataset_name}/questions_answers_YN"):
            question_files = os.listdir(f"{DATA_DIR}/{dataset_name}/questions_answers_YN")
            for question_file in question_files:
                image_file_name = os.path.join(DATA_DIR, dataset_name, "images", question_file.replace('.txt', image_suffix))
                with open(os.path.join(DATA_DIR, dataset_name, "questions_answers_YN", question_file), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            question, gt_answer = line.replace('\n', '').split('\t')
                            self.dataset.append({
                                "image_path": image_file_name,
                                "gt_answers": gt_answer,
                                "question": question
                            })
                        except:
                            pass

        else:
            question_files = glob.glob(f"{DATA_DIR}/{dataset_name}/*.txt")
            for question_file in question_files:
                image_file_name = question_file.replace(".txt", image_suffix)
                with open(question_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            question, gt_answer = line.replace('\n', '').split('\t')
                            self.dataset.append({
                                "image_path": image_file_name,
                                "gt_answers": gt_answer,
                                "question": question
                            })
                        except:
                            pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]