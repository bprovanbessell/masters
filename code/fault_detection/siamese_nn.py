# Siamese NN to perform image similarity
# Take the input of two images, and then do a comparison between them. 
# The network is composed of two identical networks, one for each input.
# The output of each network is concatenated and passed to a linear layer. 
# The output of the linear layer passed through a sigmoid function.
# `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
# This implementation varies from FaceNet as we use the `ResNet-18` model from
# `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
# In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
# based on https://github.com/pytorch/examples/blob/main/siamese_network/main.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import json
import os

from custom_dataset import SiameseDatasetSingleCategory, SiameseDatasetCatsDogs, SiameseDatasetPerObject
from trainer import train_siamese_epoch, ModelSaver
from logger import MetricLogger
from eval import evaluate_siamese


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        # Try with pretrained and non pretraind
        # change that here...
        weights = ResNet50_Weights.IMAGENET1K_V2
        # weights = ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        self.resnet_model = resnet50(weights=weights)
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze the weights, set them to be non trainable
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet_model.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights, we are using pre trained weights.
        # self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet_model(x)
        output = output.view(output.size()[0], -1)
        return output


    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
    

def train_test_category(category:str, train_model=True, load_model=False):

    batch_size = 64
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    random_seed= 42
    epochs = 10
    seed = 44

    weights = ResNet50_Weights.IMAGENET1K_V2
    # weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    cats_dogs_data_dir = '/Users/bprovan/University/dissertation/masters/code/data/archive/train'
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1_occluded = '/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded'


    # ds = SiameseDatasetCatsDogs(img_dir=cats_dogs_data_dir, transforms=preprocess)
    # ds = SiameseDatasetSingleCategory(img_dir=missing_parts_base_dir, category="KitchenPot", transforms=preprocess)
    
    ds = SiameseDatasetPerObject(img_dir=missing_parts_base_dir_v1_occluded, category=category, n=8, transforms=preprocess, train=False, train_split=0.7, seed=seed)
    test_ds = SiameseDatasetPerObject(img_dir=missing_parts_base_dir_v1_occluded, category=category, n=8, transforms=preprocess, train=False, train_split=0.7, seed=seed)
    # a fixed dataset for validation and testing 

    # Creating data indices for training and validation splits:
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices =indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=test_sampler)

    model = SiameseNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/comparison/', category + "_siamese_model_occ.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/', category + "_siamese_log_occ.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    scheduler = StepLR(optimizer, step_size=1)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_siamese_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, model_saver, metric_logger)
            scheduler.step()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_siamese(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}

    # test(model, device, test_dataloader, test_loader_len=len(test_indices), set='Test')

if __name__ == "__main__":

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                  'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                  'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                  'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                  'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']
    all_res_dict = {}

    for category in categories:
        print(category)
        # Train a model from scratch
        # train_test_category(category, train_model=True, load_model=False)
        
        res_dict = train_test_category(category, train_model=False, load_model=True)
        all_res_dict.update(res_dict)
        print("FINISHED: ", category, "\n")

        with open('logs/results/siamese_res_occ.json', 'w') as fp:
            json.dump(all_res_dict, fp)

