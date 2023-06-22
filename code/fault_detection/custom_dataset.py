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

        label = torch.tensor(label, dtype=torch.float32)
        return img, label
    
    def __len__(self):
        return len(self.imgs_paths)
    

class MissingPartDataset2Binary(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.png')

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str_base = img_path.split('/')[-1].split('_')[0]
        # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
        label_str_part_num = img_path.split('/')[-1].split('_')[1]

        # for now just binary classification
        if label_str_base == 'orig':
            label = 0
        else:
            label = 1

        label = torch.tensor(label, dtype=torch.float32)
        return img, label
    
    def __len__(self):
        return len(self.imgs_paths)
    

class MissingPartDatasetMultiClass(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.png')

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str_base = img_path.split('/')[-1].split('_')[0]
        # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
        label_str_part_num = img_path.split('/')[-1].split('_')[1]
        # for glasses it is easy, only leg 1 and leg 2
        if label_str_base == 'orig':
            label = 0
        elif label_str_part_num == "1":
            label = 1
        else:
            label = 2

        label = torch.tensor(label, dtype=torch.long)
        return img, label
    
    def __len__(self):
        return len(self.imgs_paths)
    

class CatsDogsDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.jpg')

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str = img_path.split('/')[-1].split('.')[0]

        # for now just binary classification
        if label_str == 'cat':
            # label = torch.zeros((1), dtype=torch.float32)
            label = 0
        else:
            # label = torch.ones((1), dtype=torch.float32)
            label = 1

        label = torch.tensor(label, dtype=torch.float32)
        return img, label
    
    def __len__(self):
        return len(self.imgs_paths)

# # transforms
# # 
# # # Can just use the transforms provided, probably will work better
# # Other augentation techniques??
# trainTansform = transforms.Compose([
# transforms.CenterCrop(1080),
# transforms.Resize(input_size),
# # transforms.RandomResizedCrop(config.IMAGE_SIZE),
# # transforms.RandomHorizontalFlip(),
# transforms.ToTensor(),
# preprocess,
# # transforms.Normalize(mean=config.MEAN, std=config.STD)
# ])
# valTransform = transforms.Compose([
# transforms.CenterCrop(1080),
# transforms.Resize(input_size),
# transforms.ToTensor(),
# # transforms.Normalize(mean=config.MEAN, std=config.STD)
# ])
 


if __name__ == "__main__":
    # verify the dataset
    data_dir = '/Users/bprovan/Desktop/glasses_basic/'

    data_dir = '/Users/bprovan/University/dissertation/masters/code/data/archive/train'

    input_size = 224
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

    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),                                
                                       transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])
    # ds = MissingPartDataset(data_dir, preprocess)
    cat_dog_ds = CatsDogsDataset(data_dir, transforms=preprocess)

    train_dataloader = DataLoader(cat_dog_ds, batch_size=10, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for i in range(10):
        print(train_labels)
        img = train_features[i].squeeze()
        label = train_labels[i]
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        print(f"Label: {label}")


    