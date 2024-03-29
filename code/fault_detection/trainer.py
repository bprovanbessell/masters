# The main training methods here
# Also saving models based on the evaluation results -> Refactor testing code to eval.py, and import that here
# Refactor the methods so the main method is cleaner

# We want to save the history as well!, Save based on best validation accuracy
# Save the, train loss, train accuracy, validation loss, validation accuracy

import torch
import torchmetrics

from eval import evaluate_multiview, evaluate_binary, evaluate_siamese, ModelSaver, evaluate_multiview_all
from logger import MetricLogger


def train_binary_baseline_epoch(model, device, train_loader, val_loader, optimizer, criterion, epoch, 
                                model_saver:ModelSaver=None, metric_logger:MetricLogger=None):
    model.train()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    correct = 0
    total_train_loss = 0

    for i, (x, y) in enumerate(train_loader):
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
        total_train_loss += loss
        # threshold of 0.5
        pred_class = torch.sigmoid(pred).round()
        pred_sigmoid = torch.sigmoid(pred)
        # print("logits", pred)
        # preds = torch.nn.functional.softmax(pred, dim=1)
        acc = acc_metric(pred, y)
        correct += pred_class.eq(y.view_as(pred_class)).sum().item()

        # if batch_idx % 50 == 0:
    train_acc = acc_metric.compute()
    acc_metric.reset()
    print(train_acc)
    print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {:.4f}%'.format(epoch, loss.item(), 100 * train_acc.item()))
    
    val_acc, val_loss, _, _, _ = evaluate_binary(model, device, val_loader, criterion, set="Validation")  

    model_saver.save_model(model, val_acc, epoch, optimizer)
    metric_logger.add_epoch_metrics(train_acc, loss.item(), val_acc, val_loss)


def train_multiview_epoch(model, device, train_loader, val_loader, optimizer, criterion, epoch, 
                          model_saver:ModelSaver=None, metric_logger:MetricLogger=None):
    model.train()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    correct = 0
    total_train_loss = 0

    for batch_idx, ((view_images, query_image), targets) in enumerate(train_loader):
        batch_size, n_views, C, H, W = view_images.size()
        # print(N,V,C,H,W)
        # flatten it out
        view_images = view_images.view(-1, C, H, W).to(device)

        view_images, query_image, targets = view_images.to(device), query_image.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(view_images, query_image).squeeze()

        if outputs.shape == torch.Size([]):
            outputs = outputs.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(loss.item())

        pred = torch.where(outputs > 0.5, 1, 0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        acc = acc_metric(pred, targets)
        total_train_loss += loss.item()

        # if batch_idx % 50 == 0:
    train_acc = acc_metric.compute()
    acc_metric.reset()
    print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {:.4f}%'.format(epoch, loss.item(), 100 * train_acc.item()))
    
    val_acc, val_loss, _, _, _ = evaluate_multiview(model, device, val_loader, criterion, set="Validation")  

    model_saver.save_model(model, val_acc, epoch, optimizer)
    metric_logger.add_epoch_metrics(train_acc.item(), loss.item(), val_acc, val_loss)    


def train_siamese_epoch(model, device, train_loader, val_loader, optimizer, criterion, epoch, 
                        model_saver:ModelSaver=None, metric_logger:MetricLogger=None):
    
    model.train()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    correct = 0
    total_train_loss = 0

    for batch_idx, ((images_1, images_2), targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()

        if outputs.shape == torch.Size([]):
            outputs = outputs.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        pred = torch.where(outputs > 0.5, 1, 0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        acc = acc_metric(pred, targets)
        total_train_loss += loss.item()

    train_acc = acc_metric.compute()
    acc_metric.reset()
    print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {:.4f}%'.format(epoch, loss.item(), 100 * train_acc.item()))
    
    val_acc, val_loss, _, _, _ = evaluate_siamese(model, device, val_loader, criterion, set="Validation")  

    model_saver.save_model(model, val_acc, epoch, optimizer)
    metric_logger.add_epoch_metrics(train_acc, loss.item(), val_acc, val_loss)



def train_multiview_epoch_all(model, device, train_loader, val_loader, optimizer, criterion, epoch, 
                          model_saver:ModelSaver=None, metric_logger:MetricLogger=None):
    model.train()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    correct = 0
    total_train_loss = 0

    for batch_idx, ((view_images, query_image), targets, cats) in enumerate(train_loader):
        batch_size, n_views, C, H, W = view_images.size()
        # print(N,V,C,H,W)
        # flatten it out
        view_images = view_images.view(-1, C, H, W).to(device)

        view_images, query_image, targets = view_images.to(device), query_image.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(view_images, query_image).squeeze()

        if outputs.shape == torch.Size([]):
            outputs = outputs.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(loss.item())

        pred = torch.where(outputs > 0.5, 1, 0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        acc = acc_metric(pred, targets)
        total_train_loss += loss.item()

        # if batch_idx % 50 == 0:
    train_acc = acc_metric.compute()
    acc_metric.reset()
    print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {:.4f}%'.format(epoch, loss.item(), 100 * train_acc.item()))
    
    val_acc, val_loss, _, _, _, _ = evaluate_multiview_all(model, device, val_loader, criterion, set="Validation")  

    model_saver.save_model(model, val_acc, epoch, optimizer)
    metric_logger.add_epoch_metrics(train_acc.item(), loss.item(), val_acc, val_loss)    

