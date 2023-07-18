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
import json


class SiameseUnseenModelDataset(Dataset):

    # Keep the training balanced. So for each object, 
    # sample n positive views -> for each compare it to a random (other) positive view, and a random negative view
    # sample n negative views -> for each compare it to a random positive view and a random (other) negative view


    def __init__(self, img_dir: str, category: str, n:int=12, transforms=None, data_split:str="train", train_split=0.7, seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.n = n
        self.data_split = data_split
        self.train = data_split == "train"

        data_split_json = "/Users/bprovan/University/dissertation/masters/code/data/UNSEEN_MODEL_SPLIT.json"

        instances_dict = {}
        # json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
        with open(data_split_json, "r") as f:
            instances_dict = json.load(f)

        instances_list = instances_dict[category][data_split]

        # base_dir_instance = os.path.join(img_dir, category, "*")
        # self.instance_dirs = glob.glob(base_dir_instance)
        self.instance_dirs = [os.path.join(img_dir, category, instance_id) for instance_id in instances_list]

        self.rng = np.random.default_rng(seed)

        self.test_pairs, self.test_targets = self.generate_pairs()
        self.pairs, self.targets = self.generate_pairs()
        self.reset_counter = 0
        self.reset_num = int(len(self.test_pairs)*train_split)
        # get the labels of the images paths, this is needed for the selection stage


    def __getitem__(self, index):

        if self.reset_counter == self.reset_num:
            self.reset_counter = 0
            
            if self.train:
                self.pairs, self.targets = self.generate_pairs()
            else:
                self.pairs, self.targets = self.test_pairs, self.test_targets

        img_path_1, img_path_2 = self.pairs[index]
        
        img_1 = Image.open(img_path_1).convert("RGB")
        img_2 = Image.open(img_path_2).convert("RGB")

        if self.transforms is not None:
            img_1 = self.transforms(img_1)
            img_2 = self.transforms(img_2)

        self.reset_counter += 1
        return (img_1, img_2), self.targets[index]
        
    def generate_pairs(self):

        pairs = []
        targets = []
        for index in range(len(self.instance_dirs)):
            instance_dir = self.instance_dirs[index]

            groups = {0:[],
                    1:[]}

            for img_path in glob.glob(os.path.join(instance_dir, "*.png")):

                label_str_base = img_path.split('/')[-1].split('_')[0]
                # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
                label_str_part_num = img_path.split('/')[-1].split('_')[1]

                # for now just binary classification, we only care about seperating the correct from the faulty classes
                if label_str_base == 'orig':
                    # label = torch.zeros((1), dtype=torch.float32)
                    groups[0].append(img_path)
                else:
                    groups[1].append(img_path)

            # For each positive img in the sample:
            pos_sample = self.rng.choice(len(groups[0]), size=self.n, replace=False)
            for pos_anchor_index in pos_sample:
                # get a random positive pair, that is different
                pos_index = self.rng.integers(0, len(groups[0]))

                while pos_index == pos_anchor_index:
                    pos_index = self.rng.integers(0, len(groups[0]))

                pairs.append((groups[0][pos_anchor_index], groups[0][pos_index]))
                # Add a positive label(they are the same), there should be no distance between tehm
                target = torch.tensor(0, dtype=torch.float)
                targets.append(target)

                # now get a random negative pair
            for pos_anchor_index in pos_sample:
                # get a random positive pair, that is different
                neg_index = self.rng.integers(0, len(groups[0]))

                pairs.append((groups[0][pos_anchor_index], groups[1][neg_index]))

                target = torch.tensor(1, dtype=torch.float)
                targets.append(target)

            # Could do here where we generate for randomly 
            
        return pairs, targets

    def __len__(self):
        # number of pairs, for each instance we have the number of samples, times 2 (positive and negative)
        # * self.n * 2
        # But we only want the indexes per 
        # return len(self.instance_dirs) * self.n * 2
        return len(self.test_pairs)


class ViewCombUnseenModelDataset(Dataset):
    # Try to compare the same objects only?
    # So for each object, compare it to all the other correct views (combination), and negative views.
    # A lot of examples actually. 
    # Keep the training balanced. So for each object, 
    # We get the 12 reference views -> representation of the 3D object

    # sample n positive views (different from the 12 reference views) as positive pair samples
    # sample n negative views -> for negative pair samples.

    # Our anchor (view combination) will always be the same, and then we compare many positive images, and negative images to that.


    def __init__(self, img_dir: str, category: str, n_views:int=12, n_samples:int=12,  transforms=None, data_split:str="train", train_split=0.7, seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.n_views = n_views
        self.n_samples = n_samples
        self.data_split = data_split
        self.train = data_split == "train"
            
        data_split_json = "/Users/bprovan/University/dissertation/masters/code/data/UNSEEN_MODEL_SPLIT.json"
        instances_dict = {}
        # json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
        with open(data_split_json, "r") as f:
            instances_dict = json.load(f)

        instances_list = instances_dict[category][data_split]

        # base_dir_instance = os.path.join(img_dir, category, "*")
        # self.instance_dirs = glob.glob(base_dir_instance)
        self.instance_dirs = [os.path.join(img_dir, category, instance_id) for instance_id in instances_list]

        self.rng = np.random.default_rng(seed)

        # Do this once at the start so they don't change
        self.test_pairs, self.test_targets = self.generate_pairs()
        self.pairs, self.targets = self.generate_pairs()
        self.reset_counter = 0
        self.reset_num = int(len(self.test_pairs)*train_split)

    def generate_pairs(self):
        # We need the reference views
        # We should have 24 images, so we take the 12 spaced at 30 degrees, so should be at indexes[0,2,4,6,8,10,12,14,16,18,20,22]
        # eg orig_*Index*
        pairs = []
        targets = []
        for index in range(len(self.instance_dirs)):
            # get the reference views first.
            # Actually don't we want to parameterise the reference views? For the moment it is fixed at 12
            reference_views = []
            instance_dir = self.instance_dirs[index]

            groups = {0:[],
                    1:[]}

            for img_path in glob.glob(os.path.join(instance_dir, "*.png")):

                label_str_base = img_path.split('/')[-1].split('_')[0]
                # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
                label_str_part_num = img_path.split('/')[-1].split('_')[1]

                # for now just binary classification, we only care about seperating the correct from the faulty classes
                if label_str_base == 'orig':
                    if int(label_str_part_num[0:-4]) % 2 == 0:
                        # Reference view
                        reference_views.append(img_path)
                    else:
                        groups[0].append(img_path)
                else:
                    groups[1].append(img_path)

            # Now get positive and negative samples
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

        if self.reset_counter == self.reset_num:
            self.reset_counter = 0
            print("RESET")
            
            if self.train:
                self.pairs, self.targets = self.generate_pairs()
            else:
                self.pairs, self.targets = self.test_pairs, self.test_targets

        view_img_paths, img_path_2 = self.pairs[index]
        
        view_imgs = [Image.open(view_img_path).convert("RGB") for view_img_path in view_img_paths]
        img_2 = Image.open(img_path_2).convert("RGB")

        if self.transforms is not None:
            view_imgs_2 = torch.stack([self.transforms(img) for img in view_imgs])
            img_2 = self.transforms(img_2)

        self.reset_counter += 1
        return (view_imgs_2, img_2), self.targets[index]

    def __len__(self):
        return len(self.test_pairs)
        
