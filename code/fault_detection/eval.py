import torch
import torchmetrics

def preds_y_list(dataloader, model, device):
    preds = []
    labels = []

    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device).reshape(-1, 1))
        pred = model(x)


def evaluate_binary(model, device, dataloader, criterion, set="Test"):

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    # acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='none').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)
    test_loss = 0
    total_items = 0
    correct = 0

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
            test_loss += criterion(pred, y).sum().item()
            pred_class = torch.sigmoid(pred).round()
            preds_sigmoid = torch.sigmoid(pred)
            acc = acc_metric(preds_sigmoid, y)
            # acc_sample = acc_sample_metric(preds_sigmoid, pred_class)
            prec = prec_metric(preds_sigmoid, y)
            conf = confmat(preds_sigmoid, y)
            correct += (pred_class == y).type(torch.float).sum().item()

            total_items += len(y)

        total_acc = acc_metric.compute()
        precision = prec_metric.compute()
        confusion_matrix = confmat.compute()
        test_loss /= total_items
        # total_acc_sample = acc_sample_metric.compute()
        print(set, " set:")
        print('Average loss: {:.4f}, Correct: {}/{}'.format(test_loss, correct, total_items))

        print("Accuracy: {:.4f}%, Precision: {:.4f}".format((100 * total_acc.item()), precision.item()))
        # class 0 [True positive, False negative]
        # class 1 [False Positive, True Negative]
        class0_acc = confusion_matrix[0][0]/(confusion_matrix[0][1] + confusion_matrix[0][0])
        class1_acc = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])

        print("Class 0 accuracy: {:.4f}, Class 1 accuracy: {:.4f}".format(class0_acc.item(), class1_acc.item()))
        print("Confusion matrix: ", confusion_matrix)

        return total_acc.item(), test_loss, precision.item(), class0_acc.item(), class1_acc.item()
    

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
    

def evaluate_siamese(model, device, test_loader, criterion, set='Test'):
    model.eval()
    
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    # acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='none').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)
    test_loss = 0
    total_items = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.

    with torch.no_grad():
        for ((images_1, images_2), targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            if outputs.shape == torch.Size([]):
                outputs = outputs.view(-1)
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            # print("pred", pred)
            # print("targets", targets)
            acc = acc_metric(pred, targets)
            prec = prec_metric(pred, targets)
            conf = confmat(pred, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_items += len(targets)

    test_loss /= total_items

    print(set, " set:")
    print('Average loss: {:.4f}, Correct: {}/{}\n'.format(test_loss, correct, total_items))
    
    total_acc = acc_metric.compute()
    precision = prec_metric.compute()
    confusion_matrix = confmat.compute()
    # total_acc_sample = acc_sample_metric.compute()

    print("{} Accuracy: {:.4f}, {} Precision: {:.4f}".format(set, total_acc.item(), set, precision.item()))
    class0_acc = confusion_matrix[0][0]/(confusion_matrix[0][1] + confusion_matrix[0][0])
    class1_acc = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])
    print("Class 0 Accuracy: ", class0_acc.item(), ", Class 1 Accuracy: ", class1_acc.item())
    print("Confusion matrix: ", confusion_matrix)

    return total_acc.item(), test_loss, precision.item(), class0_acc.item(), class1_acc.item()


def evaluate_multiview(model, device, test_loader, criterion, set:str="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total_items = 0

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    # acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='none').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        for ((view_images, query_images), targets) in test_loader:
            batch_size, n_views, C, H, W = view_images.size()
            # print(N,V,C,H,W)
            view_images = view_images.view(-1,C,H,W).to(device)
            view_images, query_images, targets = view_images.to(device), query_images.to(device), targets.to(device)
            outputs = model(view_images, query_images).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability

            acc = acc_metric(pred, targets)
            # acc_sample = acc_sample_metric(preds_sigmoid, pred_class)
            prec = prec_metric(pred, targets)
            conf = confmat(pred, targets)
            # print("pred", pred)
            # print("targets", targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_items += len(targets)

    test_loss /= total_items

    # With cats and dogs, can achieve 95% accuracy in the first epoch.
    # But for some reason, does not work with the Kitchen Pot category.
    print(set, " set:")
    print('Average loss: {:.4f}, Correct: {}/{}\n'.format(test_loss, correct, total_items))
    
    total_acc = acc_metric.compute()
    precision = prec_metric.compute()
    confusion_matrix = confmat.compute()
    # total_acc_sample = acc_sample_metric.compute()

    print("{} Accuracy: {:.4f}, {} Precision: {:.4f}".format(set, total_acc.item(), set, precision.item()))
    class0_acc = confusion_matrix[0][0]/(confusion_matrix[0][1] + confusion_matrix[0][0])
    class1_acc = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])
    print("Class 0 Accuracy: ", class0_acc.item(), ", Class 1 Accuracy: ", class1_acc.item())
    print("Confusion matrix: ", confusion_matrix)

    return total_acc.item(), test_loss


class ModelSaver():
    def __init__(self, save_path) -> None:
        self.best_val_acc = 0
        self.save_path = save_path

    def save_model(self, model, val_acc, epoch, optimizer):
        if val_acc > self.best_val_acc:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, self.save_path)
            
    def load_model(self, model, optimizer):

        checkpoint = torch.load(self.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        # model.eval()
        # # - or -
        # model.train()
        return model
