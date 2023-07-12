# The main training methods here
# Also saving models based on the evaluation results -> Refactor testing code to eval.py, and import that here
# Refactor the methods so the main method is cleaner

import torch
import torch.nn as nn
import torchmetrics

from eval import evaluate_multiview

def train_multiview_epoch(model, device, train_loader, val_loader, optimizer, criterion, epoch):
    model.train()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    correct = 0

    for batch_idx, ((view_images, query_image), targets) in enumerate(train_loader):
        batch_size, n_views, C, H, W = view_images.size()
        # print(N,V,C,H,W)
        # flatten it out
        view_images = view_images.view(-1, C, H, W).to(device)

        view_images, query_image, targets = view_images.to(device), query_image.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(view_images, query_image).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        pred = torch.where(outputs > 0.5, 1, 0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        acc = acc_metric(pred, targets)

        # if batch_idx % 50 == 0:
    train_acc = acc_metric.compute()
    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: '.format(epoch, loss.item(), 100 * train_acc.item()))
    
    val_acc, val_loss = evaluate_multiview(model, device, val_loader, criterion, set="Validation")        
    

    
