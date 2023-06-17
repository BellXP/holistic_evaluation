import os
# from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from typing import Any, Tuple, Sequence


class ImageNetDataset(ImageNet):

    def __init__(self, root: str='datasets/ImageNet', split: str = "val", **kwargs: Any):
        super().__init__(root, split, **kwargs)

    def __getitem__(self, index: int) -> Tuple[str, Sequence[str]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # all possible class names
        answers = self.classes[target]
        # sample = self.loader(path)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return {
            "image_path": path,
            "gt_answers": answers,
        }
