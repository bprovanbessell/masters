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
    prec_metric = torchmetrics.Precision(task='binary', average='none').to(device)

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
            acc = acc_metric(torch.sigmoid(pred), y)
            prec = prec_metric(torch.sigmoid(pred), y)
            # Correct += (pred_class == y).type(torch.float).sum().item()

        total_acc = acc_metric.compute()
        precision = prec_metric.compute()

        print("{} accuracy: {:.4f}, {} precision: {:.4f}".format(set, total_acc.item(), set, precision.item()))

        return total_acc

