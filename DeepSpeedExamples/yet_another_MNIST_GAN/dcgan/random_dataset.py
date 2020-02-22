"""A dataset which returns a random tensor."""

import torch
import torch.utils.data as data
import torchvision.datasets.vision
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def is_positive_integer(value):
    if not(isinstance(value, int)):
        return False
    try:
        tmp = int(value)
    except ValueError:
        return False
    if not value > 0:
        return False
    return True

class RandomDataset(data.Dataset):
    """RandomDataset.

     Args:
         batch_size (integer): Number of samples in the batch.
         sample_size (integer): Size of the sample.
         shape (tuple): The shape in (rows, cols) of the random data to be returned.
         transform (callable, optional): A function/transform that takes in an PIL image
             and returns a transformed version. E.g, ``transforms.RandomCrop``
         target_transform (callable, optional): A function/transform that takes in the
             target and transforms it.

    Typical usage:
        from dcgan.random_dataset import RandomDataset
        rando = random_dataset.RandomDataset(batch_size, 10)

        for i, data in enumerate(rando):
            print('Iteration: ', i)
            print('Data:      ', data)

     """
    # usually these guys have a root directory, but we're not relying on
    # any of yer actual files ...


    def __init__(self, batch_size, sample_size, transforms=None, transform=None, target_transform=None):

        if not is_positive_integer(batch_size) or not is_positive_integer(sample_size):
            raise RuntimeError('Paramaters batch_size and/or sample_size must be positive integers.' +
                               ' Both must be greater than zero')

        self.batch_size = batch_size
        self.sample_size = sample_size

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = torchvision.StandardTransform(transform, target_transform)
        self.transforms = transforms


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tensor: tensor of m random numbers.
        """
        return torch.rand(self.batch_size, self.sample_size)

    def __len__(self):
        return self.sample_size

    # def __repr__(self): ... should be inherited from the parent class
    # ... so it's worth testing to see if it works ...from