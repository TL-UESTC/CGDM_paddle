from x2paddle import torch2paddle
import paddle
from paddle.io import Dataset
from x2paddle.torch2paddle import DataLoader
from paddle.vision import transforms
from paddle.vision import datasets
import numpy as np
import os
from PIL import Image
from .folder import ImageFolder_ind
from paddle.io import RandomSampler


class OfficeAmazonDataset(Dataset):
    """Class to create an iterable dataset
    of images and corresponding labels """

    def __init__(self, image_folder_dataset, transform=None):
        super(OfficeAmazonDataset, self).__init__()
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset.imgs)

    def __getitem__(self, idx):
        img, img_label = image_folder_dataset[idx][0], image_folder_dataset[idx
            ][1]
        if self.transform is not None:
            self.transform(img)
        img_label_pair = {'image': img, 'class': img_label}
        return img_label_pair


def get_dataloader(dataset, batch_size, train_ratio=0.7):
    """
    Splits a dataset into train and test.
    Returns train_loader and test_loader.
    """

    def get_subset(indices, start, end):
        return indices[start:start + end]
    TRAIN_RATIO, VALIDATION_RATIO = train_ratio, 1 - train_ratio
    train_set_size = int(len(dataset) * TRAIN_RATIO)
    validation_set_size = int(len(dataset) * VALIDATION_RATIO)
    indices = paddle.randperm(len(dataset))
    train_indices = get_subset(indices, 0, train_set_size)
    validation_indices = get_subset(indices, train_set_size,
        validation_set_size)
    train_sampler = RandomSampler(train_indices)
    val_sampler = RandomSampler(validation_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=\
        train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=\
        val_sampler, num_workers=4)
    return train_loader, val_loader


def get_office_dataloader(name_dataset, batch_size, train=True):
    """
    Creates dataloader for the datasets in office datasetself.
    Uses get_mean_std_dataset() to compute mean and std along the
    color channels for the datasets in office.
    """
    root_dir = './dataset/office/%s/images' % name_dataset
    __datasets__ = ['amazon', 'dslr', 'webcam']
    if name_dataset not in __datasets__:
        raise ValueError('must introduce one of the three datasets in office')
    mean_std = {'amazon': {'mean': [0.7923, 0.7862, 0.7841], 'std': [0.3149,
        0.3174, 0.3193]}, 'dslr': {'mean': [0.4708, 0.4486, 0.4063], 'std':
        [0.2039, 0.192, 0.1996]}, 'webcam': {'mean': [0.6119, 0.6187, 
        0.6173], 'std': [0.2506, 0.2555, 0.2577]}}
    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
        transforms.CenterCrop(224), torch2paddle.ToTensor(), torch2paddle.
        Normalize(mean=mean_std[name_dataset]['mean'], std=mean_std[
        name_dataset]['std'])])
    dataset = ImageFolder_ind(root=root_dir, transform=data_transforms)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=\
        train, num_workers=4, drop_last=False)
    return dataset_loader
