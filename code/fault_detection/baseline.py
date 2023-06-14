"""

This will form the baseline binary fault classification.
2 main models
- Most basic one which takes in one image as imput, and trys to detect whether it has a part missing or not (fault)
- base it off https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

try this first
https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch


- feature vector stacking
One that takes in multiple images, a feature vector is taken from each of them, and they are stacked together, 
and then a classifier is trained on them.

There is also yolov6 in pytorch, https://github.com/meituan/YOLOv6/blob/main/turtorial.ipynb
Could use that as another feature extractor (This time trained on COCO), so higher resolution. (COCO 640 x 336?) something like that

there is timm for imagenet stuff
https://huggingface.co/docs/timm/quickstart, lots of already existing and pretrained models. Probably the best/easiest way. 
They include mostly updated and available models (I can actually use it, as opposed to the BASIC model.)

"""

import torch
import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np

import custom_dataset

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
data_dir = '/Users/bprovan/University/dissertation/masters/code/data/archive/train'

# pass in the transform for the pretrained model
# ds = custom_dataset.MissingPartDataset(img_dir=data_dir, transforms=preprocess)
ds = custom_dataset.CatsDogsDataset(img_dir=data_dir, transforms=preprocess)

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

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# model = resnet18(weights=ResNet18_Weights.DEFAULT)

# freeze the weights, set them to be non trainable
for param in model.parameters():
    param.requires_grad = False

modelOutputFeats = model.fc.in_features

# currently just binary classification
model.fc = nn.Linear(modelOutputFeats, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
# can set the learning rate if needed?, but otherwise us the default
optimizer = torch.optim.Adam(model.fc.parameters())

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# calculate steps per epoch for training and validation set
trainSteps = len(train_indices) // batch_size
valSteps = len(val_indices) // batch_size
print("steps", trainSteps, valSteps)

acc_metric = torchmetrics.Accuracy(task='binary').to(device)

# ------------- Training -------------
from tqdm import tqdm

epochs = 10

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
    # for (i, (x, y)) in enumerate(train_dataloader):
    # send the input to the device
        (x, y) = (x.to(device), y.to(device).reshape(-1,1))

        # also we have 
        # y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
        # y_batch = y_batch.to(device) #move to gpu

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
        acc = acc_metric(pred_sigmoid, y)
        trainCorrect += (pred_class == y).type(torch.float).sum().item()

    # trainCorrect += (pred.argmax(1) == y).type(
    # torch.float).sum().item()

    # accumulates over all batches
    train_acc = acc_metric.compute()
    acc_metric.reset()
    # validation
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in val_dataloader:
            (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
            pred = model(x)
            totalValLoss += criterion(pred, y)
            # calculate the number of correct predictions
            # threshold of 0.5
            pred_class = torch.sigmoid(pred).round()
            pred_sigmoid = torch.sigmoid(pred)
            acc = acc_metric(pred_sigmoid, y)
            valCorrect += (pred_class == y).type(torch.float).sum().item()

        val_acc = acc_metric.compute()

        print("TORCHMETRICS", train_acc, val_acc)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_indices)
        valCorrect = valCorrect / len(val_indices)
        # update our training history
        hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        hist["train_acc"].append(trainCorrect)
        hist["val_loss"].append(avgValLoss.cpu().detach().numpy())
        hist["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))

        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, valCorrect))
