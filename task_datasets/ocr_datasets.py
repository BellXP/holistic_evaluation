import os
from torch.utils.data import Dataset


class ocrDataset(Dataset):
    data_root = '/nvme/share/OCR_Datasets'

    def __init__(
        self,
        dataset_name = "ct80"
    ):
        self.dataset_name = dataset_name
        file_path = os.path.join(self.data_root, f'{dataset_name}/test_label.txt')
        file = open(file_path, "r")
        self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        img_path = self.lines[idx].split()[0]
        answers = self.lines[idx].split()[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}