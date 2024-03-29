import torch
import torchmetrics

def evaluate_binary(model, device, dataloader, criterion, set="Test"):

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    # acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='micro').to(device)
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
            # threshold of 0.5
            test_loss += criterion(pred, y).sum().item()
            pred_class = torch.sigmoid(pred).round()
            preds_sigmoid = torch.sigmoid(pred)
            acc = acc_metric(preds_sigmoid, y)
            prec = prec_metric(preds_sigmoid, y)
            conf = confmat(preds_sigmoid, y)
            correct += (pred_class == y).type(torch.float).sum().item()

            total_items += len(y)

        total_acc = acc_metric.compute()
        precision = prec_metric.compute()
        confusion_matrix = confmat.compute()
        test_loss /= total_items
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
    prec_metric = torchmetrics.Precision(task='binary', average='micro').to(device)
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
    prec_metric = torchmetrics.Precision(task='binary', average='micro').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        for ((view_images, query_images), targets) in test_loader:
            batch_size, n_views, C, H, W = view_images.size()
            # print(N,V,C,H,W)
            view_images = view_images.view(-1,C,H,W).to(device)
            view_images, query_images, targets = view_images.to(device), query_images.to(device), targets.to(device)
            outputs = model(view_images, query_images).squeeze()
            if outputs.shape == torch.Size([]):
                outputs = outputs.view(-1)
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

    return total_acc.item(), test_loss, precision.item(), class0_acc.item(), class1_acc.item()



def evaluate_multiview_all(model, device, test_loader, criterion, set:str="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total_items = 0

    {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}

    cat_label_dict_metric = {'KitchenPot': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
                    'USB': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                    'Cart': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
                    'Box': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
                    'Pliers': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                    'WashingMachine': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
                    'Lighter': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
      'Switch': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
      'Laptop': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
    'Bucket': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
      'Globe': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
        'Trashcan': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Luggage': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
              'Window': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                'Faucet': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                  'Eyeglasses': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                    'Kettle': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
                      'Toilet': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Oven': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)},
              'Stapler': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
              'Phone': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
              'Trash Can': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
              'Scissors': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
              'Dish Washer': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Lamp': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Sitting Furniture': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Table': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Storage Furniture': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}, 
            'Pot': {"acc_metric":torchmetrics.Accuracy(task='binary').to(device),
    "prec_metric":torchmetrics.Precision(task='binary', average='micro').to(device)}}

    # set up metrics
    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    # acc_sample_metric = torchmetrics.Accuracy(task='binary', multidim_average='samplewise').to(device)
    prec_metric = torchmetrics.Precision(task='binary', average='micro').to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        for ((view_images, query_images), targets, cat_labels) in test_loader:
            batch_size, n_views, C, H, W = view_images.size()
            # print(N,V,C,H,W)
            view_images = view_images.view(-1,C,H,W).to(device)
            view_images, query_images, targets = view_images.to(device), query_images.to(device), targets.to(device)
            outputs = model(view_images, query_images).squeeze()
            if outputs.shape == torch.Size([]):
                outputs = outputs.view(-1)
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

            for i, cat_label in enumerate(cat_labels):

                cat_label_dict_metric[cat_label]["acc_metric"](pred[i].reshape(1), targets[i].reshape(1))
                cat_label_dict_metric[cat_label]["prec_metric"](pred[i].reshape(1), targets[i].reshape(1))

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

    all_res_dict = {}

    for category, metrics in cat_label_dict_metric.items():
        all_res_dict[category] = {}

        for metric_name, metric_obj in metrics.items():
            if "acc" in metric_name:
                all_res_dict[category]['accuracy'] = metric_obj.compute().item()
            else:
                all_res_dict[category]['precision'] = metric_obj.compute().item()

    return total_acc.item(), test_loss, precision.item(), class0_acc.item(), class1_acc.item(), all_res_dict



class ModelSaver():
    def __init__(self, save_path) -> None:
        self.best_val_acc = 0
        self.save_path = save_path

    def save_model(self, model, val_acc, epoch, optimizer):
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc

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
        print("epoch", epoch)

        # model.eval()
        # # - or -
        # model.train()
        return model
