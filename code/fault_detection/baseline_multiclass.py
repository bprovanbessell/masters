# Baseline for (normal single view) multiclass object classification.

import custom_dataset
import eval
import models

import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch import nn
import torchmetrics
from torchmetrics.classification import BinaryF1Score
from ignite.metrics import ClassificationReport
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np

# ---- use gpu or cpu ----
device = "mps" if torch.backends.mps.is_available() \
else "gpu" if torch.cuda.is_available() else "cpu"

print(device)

weights = ResNet50_Weights.IMAGENET1K_V2
# weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

flower_data_dir = '/Users/bprovan/University/dissertation/masters/code/data/flower_photos'

ds = datasets.ImageFolder(root=flower_data_dir, transform=preprocess) # 5 classes in the flowers dataset


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
train_dataloader = DataLoader(ds, batch_size=batch_size, 
                                sampler=train_sampler)
val_dataloader = DataLoader(ds, batch_size=batch_size,
                                    sampler=val_sampler)
test_dataloader = DataLoader(ds, batch_size=batch_size,
                                    sampler=test_sampler)

print("dataloader length", len(train_dataloader))


weights = ResNet50_Weights.IMAGENET1K_V2
# weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
model = resnet50(weights=weights)
# model = resnet18(weights=ResNet18_Weights.DEFAULT)

modelOutputFeats = model.fc.in_features
# freeze the weights, set them to be non trainable
for param in model.parameters():
    param.requires_grad = False


num_classes = len(ds.classes)

model.fc = nn.Linear(modelOutputFeats, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters())

from tqdm import tqdm

acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

trainSteps = len(train_indices) // batch_size
valSteps = len(val_indices) // batch_size

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

        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        preds = torch.nn.functional.softmax(pred, dim=1)
        acc = acc_metric(pred, y)
        
        # if i % 50 == 0:

            # print("softmax preds", preds)
            # print(y)
            # preds = nn.LogSoftmax(pred)

            # print(y)
            # print(pred)
            # Its fecking bugged on M1
            # print("argmax", torch.max(pred, dim=1).indices)    
            # print(trainCorrect)
            
        trainCorrect += ((torch.max(pred, dim=1).indices) == y).type(torch.float).sum().item()

    # accumulates over all batches
    train_acc = acc_metric.compute()
    acc_metric.reset()
    # validation
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in val_dataloader:
            (x, y) = (x.to(device), y.to(device))
            # (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
            pred = model(x)
            totalValLoss += criterion(pred, y)
            # calculate the number of correct predictions
            # threshold of 0.5
            # pred_class = torch.sigmoid(pred).round()
            # pred_sigmoid = torch.sigmoid(pred)
            preds = torch.nn.functional.softmax(pred, dim=1)
            acc = acc_metric(preds, y)
            valCorrect += ((torch.max(pred, dim=1).indices) == y).type(torch.float).sum().item()

        val_acc = acc_metric.compute()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        avgTrainAcc = trainCorrect/len(train_indices)
        avgValAcc = valCorrect/len(val_indices)
        # update our training history
        # hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        # hist["train_acc"].append(train_acc.item())
        # hist["val_loss"].append(avgValLoss.cpu().detach().numpy())
        # hist["val_acc"].append(val_acc.item())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, train_acc.item()))
        print(avgTrainAcc)

        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, val_acc.item()))
        print(avgValAcc)
