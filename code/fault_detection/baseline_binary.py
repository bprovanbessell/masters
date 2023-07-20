"""
Binary classifier baseline 
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import numpy as np
import os
from tqdm import tqdm

import models
from eval import evaluate_binary, ModelSaver
from trainer import train_binary_baseline_epoch
from logger import MetricLogger

from custom_dataset import MissingPartDatasetBinary, MissingPartDatasetBalancedBinary
from torch.utils.data.sampler import SubsetRandomSampler
import json

def train_test_category(category:str, train_model=True, load_model=False):

    batch_size = 32
    validation_split = 0.2
    test_split = 0.2
    shuffle_dataset = True
    seed = 44
    epochs = 20

    weights = ResNet50_Weights.IMAGENET1K_V2
    # weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)

    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'

    ds = MissingPartDatasetBalancedBinary(img_dir_base='/Users/bprovan/University/dissertation/datasets/images_ds_v2/query_images', category=category, transforms=preprocess, seed=seed)
    test_ds = MissingPartDatasetBalancedBinary(img_dir_base='/Users/bprovan/University/dissertation/datasets/images_ds_v2/query_images/test', category=category, transforms=preprocess, seed=seed)
    val_ds = MissingPartDatasetBalancedBinary(img_dir_base='/Users/bprovan/University/dissertation/datasets/images_ds_v2/query_images/validation', category=category, transforms=preprocess, seed=seed)


    dataset_size = len(ds)
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=None)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=None)
    
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    
    num_classes = 1

    model = models.resnet50_pretrained_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters())

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/baseline_binary/', category + "_binary_model2.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/', category + "_binary_log2.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_binary_baseline_epoch(model, device, train_dataloader, val_dataloader,
                                         optimizer, criterion, epoch, model_saver, metric_logger)

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_binary(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}


if __name__ == "__main__":

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                  'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                  'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                  'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                  'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']

    # category = 'USB'
    # train_category(category)
    train_test_category(category='KitchenPot', train_model=True, load_model=False)

    all_res_dict = {}

    # for category in categories:
    #     print(category)
    #     # Train a model from scratch
    #     # train_test_category(category, train_model=True, load_model=False)
        
    #     res_dict = train_test_category(category, train_model=False, load_model=True)
    #     all_res_dict.update(res_dict)
    #     print("FINISHED: ", category, "\n")

    #     with open('logs/baseline_binary.json', 'w') as fp:
    #         json.dump(all_res_dict, fp)
