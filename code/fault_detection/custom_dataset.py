import os
import numpy as np
import torch
from PIL import Image
import glob

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import random

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

# Map the metadata to actual labels
label_dict = {}
label_dict['true'] = 0
label_dict['part1'] = 1
label_dict['part2'] = 1

class MissingPartDataset(Dataset):
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
    

class MissingPartDataset2Binary(Dataset):
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
    

class MissingPartDatasetMultiClass(Dataset):
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
    

class CatsDogsDataset(Dataset):

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


class SiameseDatasetSingleCategory(Dataset):

    def __init__(self, img_dir, category, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        base_dir = os.path.join(img_dir, category, "*", "*.png")
        self.imgs_paths = glob.glob(base_dir)
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str = img_path.split('/')[-1].split('_')[0]

            # for now just binary classification, we only care about seperating the correct from the faulty classes
            if label_str == 'orig':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
            else:
                self.groups[1].append(img_path)


    def __getitem__(self, idx):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """
        selected_class = random.randint(0, 1)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.groups[selected_class]) -1)
        
        # pick the index to get the first image
        img_path_1 = self.groups[selected_class][random_index_1]

        # get the first image
        img_1 = Image.open(img_path_1).convert("RGB")


        # same class
        if idx % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.groups[selected_class]) -1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.groups[selected_class]) -1)
            
            # pick the index to get the second image
            img_path_2 = self.groups[selected_class][random_index_2]

            # get the second image
            img_2 = Image.open(img_path_2).convert("RGB")

            # set the label for this example to be positive (1), similarity of 1
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 1)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 1)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.groups[other_selected_class])-1)

            # pick the index to get the second image
            img_path_2 = self.groups[other_selected_class][random_index_2]

            # get the second image
            img_2 = Image.open(img_path_2).convert("RGB")

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        if self.transforms is not None:
            img_1 = self.transforms(img_1)
            img_2 = self.transforms(img_2)

        return img_1, img_2, target

    
    def __len__(self):
        return len(self.imgs_paths)
    

 


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


    