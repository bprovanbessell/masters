"""

This will form the baseline binary fault classification.
2 main models
- Most basic one which takes in one image as imput, and trys to detect whether it has a part missing or not (fault)
- base it off https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

there is timm for imagenet stuff
https://huggingface.co/docs/timm/quickstart, lots of already existing and pretrained models. Probably the best/easiest way. 
They include mostly updated and available models (I can actually use it, as opposed to the BASIC model.) Could be better
to use incpetion, or efficientnet or something...

"""

import torch
import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np

import custom_dataset
import eval
import models

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch import nn
import torchmetrics
from torchmetrics.classification import BinaryF1Score
from ignite.metrics import ClassificationReport

weights = ResNet50_Weights.IMAGENET1K_V2
# weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()


device = "mps" if torch.backends.mps.is_available() \
else "gpu" if torch.cuda.is_available() else "cpu"

print(device)

# for baseline, use pretrained on imagenet, binary classification. 

# Train, test, validation split
# For initial baseline train test split, we can just randomly split up the images from the file paths.

# so our initial size of photo is 1920x1080, we probably need to change this at some point

input_size = 224

# data_dir = '/Users/bprovan/Desktop/glasses_basic/'
cats_dogs_data_dir = '/Users/bprovan/University/dissertation/masters/code/data/archive/train'
flower_data_dir = '/Users/bprovan/University/dissertation/masters/code/data/flower_photos'
data_dir = '/Users/bprovan/Desktop/gen_images_640/'
data_dir = '/Users/bprovan/Desktop/glasses_640/'
missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0/Eyeglasses/'

# pass in the transform for the pretrained model
# ds = custom_dataset.MissingPartDataset(img_dir=data_dir, transforms=preprocess)
# ds = custom_dataset.CatsDogsDataset(img_dir=cats_dogs_data_dir, transforms=preprocess)
ds = custom_dataset.MissingPartDataset2Binary(img_dir=missing_parts_base_dir, transforms=preprocess)
# ds = custom_dataset.MissingPartDatasetMultiClass(img_dir=data_dir, transforms=preprocess)

# ds = datasets.ImageFolder(root=flower_data_dir, transform=preprocess) # 5 classes in the flowers dataset

batch_size = 32
validation_split = 0.2
test_split = 0.2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(ds)
indices = list(range(dataset_size))
val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

print(val_split_index)
print(test_split_index)
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
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
val_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                    sampler=val_sampler)
test_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                    sampler=test_sampler)

print("dataloader length", len(train_dataloader))

# ------- Model setup -------

num_classes = 1

model = models.resnet50_pretrained_model(num_classes=num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# and then for prediction
# torch.nn.functional.softmax(output[0], dim=0)

# can set the learning rate if needed?, but otherwise us the default
optimizer = torch.optim.Adam(model.fc.parameters())

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# calculate steps per epoch for training and validation set
trainSteps = len(train_indices) // batch_size
valSteps = len(val_indices) // batch_size
print("steps", trainSteps, valSteps)

acc_metric = torchmetrics.Accuracy(task='binary').to(device)
# acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

# ------------- Training -------------
from tqdm import tqdm

epochs = 20

for epoch in tqdm(range(epochs)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (i, (x, y)) in enumerate(train_dataloader):
    # send the input to the device
        # print(y)
        # For binary
        (x, y) = (x.to(device), y.to(device).reshape(-1,1))
        # (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        # threshold of 0.5
        pred_class = torch.sigmoid(pred).round()
        pred_sigmoid = torch.sigmoid(pred)
        # print("logits", pred)
        # preds = torch.nn.functional.softmax(pred, dim=1)
        acc = acc_metric(pred, y)
        

    # accumulates over all batches
    train_acc = acc_metric.compute()
    acc_metric.reset()
    # validation
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in val_dataloader:
            # (x, y) = (x.to(device), y.to(device))
            (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
            pred = model(x)
            totalValLoss += criterion(pred, y)
            # calculate the number of correct predictions
            # threshold of 0.5
            pred_class = torch.sigmoid(pred).round()
            # pred_sigmoid = torch.sigmoid(pred)
            # preds = torch.nn.functional.softmax(pred, dim=1)
            acc = acc_metric(pred, y)

        val_acc = acc_metric.compute()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # update our training history
        hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        hist["train_acc"].append(train_acc.item())
        hist["val_loss"].append(avgValLoss.cpu().detach().numpy())
        hist["val_acc"].append(val_acc.item())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, train_acc.item()))

        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, val_acc.item()))

# ----- Test set evaluation ------

eval.evaluate_binary(test_dataloader, model=model, device=device)
# eval.evaluate_multiclass(3, test_dataloader, model=model, device=device)


