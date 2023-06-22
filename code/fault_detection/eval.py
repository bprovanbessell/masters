import torch
import torchmetrics

def preds_y_list(dataloader, model, device):
    preds = []
    labels = []

    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
        pred = model(x)


def evaluate_binary(dataloader, model, device, set="Test"):

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='none').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in dataloader:
            (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
            pred = model(x)
            # totalValLoss += criterion(pred, y)
            # calculate the number of correct predictions
            # threshold of 0.5
            pred_class = torch.sigmoid(pred).round()
            preds_sigmoid = torch.sigmoid(pred)
            acc = acc_metric(preds_sigmoid, y)
            # acc_sample = acc_sample_metric(preds_sigmoid, pred_class)
            prec = prec_metric(preds_sigmoid, y)
            conf = confmat(preds_sigmoid, y)
            # Correct += (pred_class == y).type(torch.float).sum().item()

        total_acc = acc_metric.compute()
        precision = prec_metric.compute()
        confusion_matrix = confmat.compute()
        # total_acc_sample = acc_sample_metric.compute()

        print("{} accuracy: {:.4f}, {} precision: {:.4f}".format(set, total_acc.item(), set, precision.item()))
        # print("Accuracies per class: ", total_acc_sample)
        # class 0 [True positive, False negative]
        # class 1 [False Positive, True Negative]
        class0_acc = confusion_matrix[0][0]/(confusion_matrix[0][1] + confusion_matrix[0][0])
        class1_acc = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])

        print("Class 0 accuracy: ", class0_acc.item(), "Class 1 accuracy: ", class1_acc.item())
        print("Confusion matrix: ", confusion_matrix)

        return total_acc

def evaluate_multiclass(num_classes: int, dataloader, model, device, set="Test"):

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    acc_sample_metric = torchmetrics.Accuracy(task='multiclass', multidim_average='samplewise', num_classes=num_classes).to(device)
    prec_metric = torchmetrics.Precision(task='multiclass', average='none', num_classes=num_classes).to(device)
    confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)

    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in dataloader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            # totalValLoss += criterion(pred, y)
            preds = torch.nn.functional.softmax(pred, dim=1)
            acc = acc_metric(preds, y)
            # acc = acc_metric(preds_sigmoid, y)
            acc_sample = acc_sample_metric(preds, y)
            prec = prec_metric(preds, y)
            conf = confmat(preds, y)
            # Correct += (pred_class == y).type(torch.float).sum().item()

        total_acc = acc_metric.compute()
        precision = prec_metric.compute()
        confusion_matrix = confmat.compute()
        total_acc_sample = acc_sample_metric.compute()

        print("{} accuracy: {:.4f}, {} precision: {:.4f}".format(set, total_acc.item(), set, precision.item()))
        print("Accuracies per class: ", total_acc_sample)
        # class 0 [True positive, False negative]
        # class 1 [False Positive, True Negative]

        # This is gonna have to change, but
        class0_acc = confusion_matrix[0][0]/(confusion_matrix[0][1] + confusion_matrix[0][0])
        class1_acc = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])

        # print("Class 0 accuracy: ", class0_acc.item(), "Class 1 accuracy: ", class1_acc.item())
        print("Confusion matrix: ", confusion_matrix)

        return total_acc