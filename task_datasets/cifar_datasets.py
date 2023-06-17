import os
# from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from typing import Any, Tuple, Sequence


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root: str='datasets', train: bool=False, **kwargs: Any):
        super().__init__(root, train, **kwargs)

    def __getitem__(self, index: int) -> Tuple[str, Sequence[str]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # HxWxC
        img, target = self.data[index], self.targets[index]
        # all possible class names
        answers = self.classes[target]

        return {
            "image_path": img,
            "gt_answers": answers,
        }

class CIFAR100Dataset(CIFAR10Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
