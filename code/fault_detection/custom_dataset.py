import os
import numpy as np
import torch
from PIL import Image
import glob

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

# Map the metadata to actual labels
label_dict = {}
label_dict['true'] = 0
label_dict['part1'] = 1
label_dict['part2'] = 1

class MissingPartDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.png')

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str = img_path.split('/')[-2].split('_')[-1]

        # for now just binary classification
        if label_str == 'true':
            label = 0
        else:
            label = 1

        return img, label
    
    def __len__(self):
        return len(self.imgs_paths)
    
if __name__ == "__main__":
    # verify the dataset
    data_dir = '/Users/bprovan/Desktop/glasses_basic/'

    input_size = 284
    # Other augentation techniques??
    trainTansform = transforms.Compose([
        transforms.CenterCrop(1080),
        transforms.Resize(input_size),
        # transforms.RandomResizedCrop(config.IMAGE_SIZE),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        preprocess
    # transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    ds = MissingPartDataset(data_dir, trainTansform)

    train_dataloader = DataLoader(ds, batch_size=10, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for i in range(10):
        img = train_features[i].squeeze()
        label = train_labels[i]
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        print(f"Label: {label}")


    