"""
Log metrics during training, dafault is train acc, train loss, val acc, val loss
"""

class MetricLogger():
    def __init__(self, save_path) -> None:
        self.metrics = {'train_acc': [], 
                        'train_loss': [],
                        'val_acc': [],
                        'val_loss': [],}

    def save_metrics(self):
        # serialise to json and save
        pass

    def load(self):
        pass

    def add_epoch_metrics(self, train_acc, train_loss, val_acc, val_loss):
        self.metrics['train_acc'].append(train_acc)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['val_loss'].append(val_loss)