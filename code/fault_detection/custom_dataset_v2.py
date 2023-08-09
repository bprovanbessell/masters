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



class ViewCombDatasetv2(Dataset):
    # Try to compare the same objects only?
    # So for each object, compare it to all the other correct views (combination), and negative views.
    # A lot of examples actually. 
    # Keep the training balanced. So for each object, 
    # We get the 12 reference views -> representation of the 3D object (anchor images)

    # and then we have the query images -> set out in test, validation, anf the rest

    def __init__(self, img_dir: str, category: str,  transforms=None, split:str='train', seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.split = split

        if split == "train":
            base_dir_query_instance = os.path.join(img_dir, 'query_images/train', category, "*")
            self.n_samples=9

        elif split == 'validation':
            base_dir_query_instance = os.path.join(img_dir, 'query_images/validation', category, "*")
            self.n_samples = 1

        elif split == 'test':
            base_dir_query_instance = os.path.join(img_dir, 'query_images/test', category, "*")
            self.n_samples = 2
        else:
            print("defaulting to training set")
            self.split = 'train'
            base_dir_query_instance = os.path.join(img_dir, category, "*")

        self.instance_query_dirs = glob.glob(base_dir_query_instance)
        base_dir_reference_instance = os.path.join('/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded', category, "*")
        self.instance_reference_dirs = glob.glob(base_dir_reference_instance)

        self.rng = np.random.default_rng(seed)

        self.pairs, self.targets = self.generate_pairs()


    def generate_pairs(self):
        # We need the reference views
        # We should have 24 images, so we take the 12 spaced at 30 degrees, so should be at indexes[0,2,4,6,8,10,12,14,16,18,20,22]
        # eg orig_*Index*

        pairs = []
        targets = []
        for index in range(len(self.instance_reference_dirs)):
            # get the reference views first.

            ref_instance_dir = self.instance_reference_dirs[index]
            reference_views = glob.glob(os.path.join(ref_instance_dir, "orig*.png"))

            query_instance_dir = self.instance_query_dirs[index]

            groups = {0:[],
                    1:[]}

            for img_path in glob.glob(os.path.join(query_instance_dir, "*.png")):

                label_str_base = img_path.split('/')[-1].split('_')[0]

                if label_str_base == 'orig':
                        groups[0].append(img_path)
                else:
                    groups[1].append(img_path)

            # Now get positive and negative samples
            if len(groups[1]) == 0:
                continue
            pos_sample = self.rng.choice(len(groups[0]), size=self.n_samples, replace=False)
            neg_sample = self.rng.choice(len(groups[1]), size=self.n_samples, replace=False)

            # Pair for each sample
            for pos_index in pos_sample:

                pairs.append((reference_views, groups[0][pos_index]))
                # Add a positive label(they are the same), there should be no distance between tehm
                target = torch.tensor(0, dtype=torch.float)
                targets.append(target)
            
            for neg_index in neg_sample:

                pairs.append((reference_views, groups[1][neg_index]))
                # Add a positive label(they are the same), there should be no distance between tehm
                target = torch.tensor(1, dtype=torch.float)
                targets.append(target)

        return pairs, targets

    def __getitem__(self, index):

        if self.split == 'train' and index == 0:
            self.pairs, self.targets = self.generate_pairs()
            print("gen")

        view_img_paths, img_path_2 = self.pairs[index]
        
        view_imgs = [Image.open(view_img_path).convert("RGB") for view_img_path in view_img_paths]
        img_2 = Image.open(img_path_2).convert("RGB")

        if self.transforms is not None:
            view_imgs_2 = torch.stack([self.transforms(img) for img in view_imgs])
            img_2 = self.transforms(img_2)

        return (view_imgs_2, img_2), self.targets[index]

    def __len__(self):
        return len(self.pairs)
    

# unseen anomaly categories: 

class ViewCombDatasetUnseenAnomaly(Dataset):
    # Try to compare the same objects only?
    # So for each object, compare it to all the other correct views (combination), and negative views.
    # A lot of examples actually. 
    # Keep the training balanced. So for each object, 
    # We get the 12 reference views -> representation of the 3D object (anchor images)

    # and then we have the query images -> set out in test, validation, anf the rest

    def __init__(self, img_dir: str, category: str,  transforms=None, split:str='train', seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.split = split

        if split == "train":
            self.base_dir_query = os.path.join(img_dir, 'train', category)
            self.n_samples=9
            self.instances = os.listdir(self.base_dir_query)

        elif split == 'validation':
            self.base_dir_query = os.path.join(img_dir, 'validation', category)
            self.n_samples = 9
            self.instances = os.listdir(self.base_dir_query)

        elif split == 'test':
            self.base_dir_query = os.path.join(img_dir, 'test', category)
            self.n_samples = 9
            self.instances = os.listdir(self.base_dir_query)
        else:
            print("defaulting to training set")
            self.split = 'train'
            self.base_dir_query = os.path.join(img_dir, category)
            self.instances = os.listdir(self.base_dir_query)

        # self.instance_query_dirs = glob.glob(base_dir_query_instance)
        self.base_dir_reference = os.path.join('/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded', category)
        # self.instance_reference_dirs = glob.glob(base_dir_reference_instance)

        self.instances = [x for x in self.instances if x != '.DS_Store']
        print(self.instances)

        self.rng = np.random.default_rng(seed)

        self.pairs, self.targets = self.generate_pairs()


    def generate_pairs(self):
        # We need the reference views
        # We should have 24 images, so we take the 12 spaced at 30 degrees, so should be at indexes[0,2,4,6,8,10,12,14,16,18,20,22]
        # eg orig_*Index*

        pairs = []
        targets = []
        for instance in self.instances:
            # get the reference views first.

            # ref_instance_dir = self.instance_reference_dirs[index]
            reference_views = glob.glob(os.path.join(self.base_dir_reference, instance, "orig*.png"))

            # query_instance_dir = self.instance_query_dirs[index]

            groups = {0:[],
                    1:[]}

            for img_path in glob.glob(os.path.join(self.base_dir_query, instance, "*.png")):

                label_str_base = img_path.split('/')[-1].split('_')[0]

                if label_str_base == 'orig':
                        groups[0].append(img_path)
                else:
                    groups[1].append(img_path)

            # Now get positive and negative samples
            if len(groups[1]) == 0:
                continue

            max_sample_size = min(len(groups[0]), len(groups[1]))

            if max_sample_size >= self.n_samples:
                sample_size = self.n_samples

            else:
                sample_size = max_sample_size
            pos_sample = self.rng.choice(len(groups[0]), size=sample_size, replace=False)
            neg_sample = self.rng.choice(len(groups[1]), size=sample_size, replace=False)

            # Pair for each sample
            for pos_index in pos_sample:

                pairs.append((reference_views, groups[0][pos_index]))
                # Add a positive label(they are the same), there should be no distance between tehm
                target = torch.tensor(0, dtype=torch.float)
                targets.append(target)
            
            for neg_index in neg_sample:

                pairs.append((reference_views, groups[1][neg_index]))
                # Add a positive label(they are the same), there should be no distance between tehm
                target = torch.tensor(1, dtype=torch.float)
                targets.append(target)

        return pairs, targets

    def __getitem__(self, index):

        if self.split == 'train' and index == 0:
            self.pairs, self.targets = self.generate_pairs()
            print("gen")

        view_img_paths, img_path_2 = self.pairs[index]
        
        view_imgs = [Image.open(view_img_path).convert("RGB") for view_img_path in view_img_paths]
        img_2 = Image.open(img_path_2).convert("RGB")

        if self.transforms is not None:
            view_imgs_2 = torch.stack([self.transforms(img) for img in view_imgs])
            img_2 = self.transforms(img_2)

        return (view_imgs_2, img_2), self.targets[index]

    def __len__(self):
        return len(self.pairs)
    

# unseen anomaly categories: 
        
        
