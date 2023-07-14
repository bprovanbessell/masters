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


class MissingPartDatasetBinary(Dataset):
    def __init__(self, img_dir_base, category, transforms):
        
        self.transforms = transforms

        img_dir = os.path.join(img_dir_base, category, '*/' '*.png')
        self.imgs_paths = glob.glob(img_dir)

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
    

class MissingPartDatasetBalancedBinary(Dataset):

    def __init__(self, img_dir_base, category, transforms, seed=42):
        self.img_dir = img_dir_base

        self.transforms = transforms
        self.rng = np.random.default_rng(seed)

        img_dir = os.path.join(img_dir_base, category, '*/' '*.png')
        self.imgs_paths = glob.glob(img_dir)
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str_base = img_path.split('/')[-1].split('_')[0]
            # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
            label_str_part_num = img_path.split('/')[-1].split('_')[1]

            # for now just binary classification, we only care about seperating the correct from the faulty classes
            if label_str_base == 'orig':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
            else:
                self.groups[1].append(img_path)

        # We want the same number of faulty samples as non faulty samples
        # faulty samples will be the same length (no problem) or greater length
        if len(self.groups[1]) > len(self.groups[0]):
            neg_samples_index = self.rng.choice(len(self.groups[1]), size=len(self.groups[0]), replace=False)
            neg_samples = [self.groups[1][ind] for ind in neg_samples_index]

            self.imgs_paths = self.groups[0] + neg_samples

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str_base = img_path.split('/')[-1].split('_')[0]
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

    
class SiameseDatasetCatsDogs(Dataset):

    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.jpg')
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str = img_path.split('/')[-1].split('.')[0]

            # for now just binary classification
            if label_str == 'cat':
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

        return (img_1, img_2), target

    
    def __len__(self):
        return len(self.imgs_paths)
    

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
            label_str_base = img_path.split('/')[-1].split('_')[0]
            # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
            label_str_part_num = img_path.split('/')[-1].split('_')[1]

            # for now just binary classification, we only care about seperating the correct from the faulty classes
            if label_str_base == 'orig':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
            else:
                self.groups[1].append(img_path)

        print(len(self.groups[0]))
        print(len(self.groups[0]))

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

        return (img_1, img_2), target

    
    def __len__(self):
        return len(self.imgs_paths)
    

class SiameseDatasetPerObject(Dataset):

    # Try to compare the same objects only? Will this help it at all?
    # So for each object, compare it to all the other correct views, and negative views.
    # A lot of examples actually. 
    # Keep the training balanced. So for each object, 
    # sample n positive views -> for each compare it to a random (other) positive view, and a random negative view
    # sample n negative views -> for each compare it to a random positive view and a random (other) negative view


    def __init__(self, img_dir: str, category: str, n:int=12, transforms=None, train:bool=True, train_split=0.7, seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.n = n
        self.train = train

        base_dir_instance = os.path.join(img_dir, category, "*")
        self.instance_dirs = glob.glob(base_dir_instance)
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
    

class ViewCombDataset(Dataset):
    # Try to compare the same objects only?
    # So for each object, compare it to all the other correct views (combination), and negative views.
    # A lot of examples actually. 
    # Keep the training balanced. So for each object, 
    # We get the 12 reference views -> representation of the 3D object

    # sample n positive views (different from the 12 reference views) as positive pair samples


    # sample n negative views -> for negative pair samples.

    # Our anchor (view combination) will always be the same, and then we compare many positive images, and negative images to that.


    def __init__(self, img_dir: str, category: str, n_views:int=12, n_samples:int=12,  transforms=None, train:bool=True, train_split=0.7, seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.n_views = n_views
        self.n_samples = n_samples
        self.train = train

        base_dir_instance = os.path.join(img_dir, category, "*")
        self.instance_dirs = glob.glob(base_dir_instance)

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
        

class ViewCombDifferenceDataset(Dataset):
    # For Difference Learning
    # So positive pairs are for objects in the same format
    # So we need to split the dataset per class


    def __init__(self, img_dir: str, category: str, n_views:int=12, n_samples:int=12,  transforms=None, train:bool=True, train_split=0.7, seed:int=42):
        self.img_dir = img_dir

        self.transforms = transforms
        self.n_views = n_views
        self.n_samples = n_samples
        self.train = train

        base_dir_instance = os.path.join(img_dir, category, "*")
        self.instance_dirs = glob.glob(base_dir_instance)

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

        # for each object, seperate it out into its classes
        pairs = []
        targets = []
        for index in range(len(self.instance_dirs)):

            instance_dir = self.instance_dirs[index]
            # first seperate the images into their respective classes

            class_mapper = {"orig": 0}
            class_counter = 1

            classes_dict = {"orig": []}

            for img_path in glob.glob(os.path.join(instance_dir, "*.png")):

                label_str_base = img_path.split('/')[-1].split('_')[0]
                label_str_part_num = img_path.split('/')[-1].split('_')[1]

                if label_str_base == "orig":
                    classes_dict[label_str_base].append(img_path)

                else:
                    base_obj_str = label_str_base + label_str_part_num

                    if base_obj_str not in class_mapper:
                        class_mapper[base_obj_str] = class_counter
                        class_counter += 1
                        classes_dict[base_obj_str] = []

                    classes_dict[base_obj_str].append(img_path)

            
            # now iterate through each of the classes. For each of them, we create the 3D object representation, and the negative and positive query images
            for class_name, item in classes_dict.items():
                reference_views = []
                # Create the groups
                groups = {0:[],
                          1:[]}

                # create the reference views for this class
                for img_path in item:
                    view_num = int(img_path.split('/')[-1].split('_')[-1][0:-4])

                    if view_num % 2 == 0:
                        # Reference view
                        reference_views.append(img_path)
                    else:
                        # Otherwise it is just a random positive view
                        groups[0].append(img_path)

                # All possible negative views (groups of 1) can be any other classes
                for class_name_2, items_2 in classes_dict.items():
                    if class_name_2 != class_name:
                        groups[1].extend(items_2)

                # Now generate the negative and positive pairs as before
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

        print(pairs[0], targets[0])
        print(pairs[13], targets[13])

        print(len(pairs), len(targets))
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

  
class TripletDatasetCatsDogs(Dataset):

    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.jpg')
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}
        self.labels = []

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str = img_path.split('/')[-1].split('.')[0]

            # for now just binary classification
            if label_str == 'cat':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
                self.labels.append(0)
            else:
                self.groups[1].append(img_path)
                self.labels.append(1)


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

            Randomly
        """

        anchor_path, label = self.imgs_paths[idx], self.labels[idx]
        positive_label = label
        negative_label = abs(label-1)

        # get the anchor image
        anchor_img = Image.open(anchor_path).convert("RGB")

        # get random positive image
        random_index_pos = random.randint(0, len(self.groups[positive_label]) -1)
            
        # ensure that the index of the second image isn't the same as the first image
        while self.groups[positive_label][random_index_pos] == anchor_path:
            random_index_pos = random.randint(0, len(self.groups[positive_label]) -1)
        
        # pick the index to get the second image
        img_path_pos = self.groups[positive_label][random_index_pos]
        img_pos = Image.open(img_path_pos).convert("RGB")

        # get random negative image
        random_index_neg = random.randint(0, len(self.groups[negative_label]) -1)
        
        # pick the index to get the second image
        img_path_neg = self.groups[negative_label][random_index_neg]

        # get the second image
        img_neg = Image.open(img_path_neg).convert("RGB")

        if self.transforms is not None:
            anchor_img = self.transforms(anchor_img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        return anchor_img, img_pos, img_neg

    
    def __len__(self):
        return len(self.imgs_paths)
   

class TripletDatasetMissingParts(Dataset):

    def __init__(self, img_dir, category, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        base_dir = os.path.join(img_dir, category, "*", "*.png")
        self.imgs_paths = glob.glob(base_dir)
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}
        self.labels = []

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str_base = img_path.split('/')[-1].split('_')[0]
            # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
            label_str_part_num = img_path.split('/')[-1].split('_')[1]

            # for now just binary classification, we only care about seperating the correct from the faulty classes
            if label_str_base == 'orig':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
                self.labels.append(0)
            else:
                self.groups[1].append(img_path)
                self.labels.append(1)


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

            Randomly
        """

        anchor_path, label = self.imgs_paths[idx], self.labels[idx]
        positive_label = label
        negative_label = abs(label-1)

        # get the anchor image
        anchor_img = Image.open(anchor_path).convert("RGB")

        # get random positive image
        random_index_pos = random.randint(0, len(self.groups[positive_label]) -1)
            
        # ensure that the index of the second image isn't the same as the first image
        while self.groups[positive_label][random_index_pos] == anchor_path:
            random_index_pos = random.randint(0, len(self.groups[positive_label]) -1)
        
        # pick the index to get the second image
        img_path_pos = self.groups[positive_label][random_index_pos]
        img_pos = Image.open(img_path_pos).convert("RGB")

        # get random negative image
        random_index_neg = random.randint(0, len(self.groups[negative_label]) -1)
        
        # pick the index to get the second image
        img_path_neg = self.groups[negative_label][random_index_neg]

        # get the second image
        img_neg = Image.open(img_path_neg).convert("RGB")

        if self.transforms is not None:
            anchor_img = self.transforms(anchor_img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        return anchor_img, img_pos, img_neg

    
    def __len__(self):
        return len(self.imgs_paths)
    

if __name__ == "__main__":

    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
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


    