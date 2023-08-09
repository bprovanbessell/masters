import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from custom_dataset import ViewCombDataset, ViewCombDifferenceDataset, ViewCombDatasetAllCats
from custom_dataset_v2 import ViewCombDatasetv2, ViewCombDatasetUnseenAnomaly
from unseen_model_anomoly_dataset import ViewCombUnseenModelDataset
from trainer import train_multiview_epoch, train_multiview_epoch_all
from eval import evaluate_multiview, ModelSaver, evaluate_multiview_all
from logger import MetricLogger
import os
import json

class ViewCombNetwork(nn.Module):
    def __init__(self, n_views=12, pretrained=True):
        super(ViewCombNetwork, self).__init__()

        self.n_views = n_views
        # get resnet model
        # Try with pretrained and non pretraind
        # change that here...
        # Try a more basic resnet18, cheaper
        # weights = ResNet50_Weights.IMAGENET1K_V2
        weights = ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()

        if not pretrained:
            self.resnet_model = resnet18()
        else:
            self.resnet_model = resnet18(weights=weights)
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)

        model_list = list(self.resnet_model.children())
        # print(model_list)
        # print(len(model_list))

        finetune = True
        # freeze the weights, set them to be non trainable
        # Does this work appropriately
        if finetune:
            for layer in model_list[0:-3]: 
                for param in layer.parameters():
                        param.requires_grad = False
            # pass
            # pretrained_model_except_last_layer = list(pretrained_model.children())[:-1]
            # we must be somehow able to train just the last 3 layers or something?
        else:
            for param in self.resnet_model.parameters():
                param.requires_grad = False

        self.fc_in_features = self.resnet_model.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

        # We want to reduce this to the number of features we have
        self.view_comb = nn.Sequential(
            nn.Linear(self.fc_in_features * self.n_views, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024, affine=False),
            nn.Linear(1024, self.fc_in_features),
        )

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256, affine=False),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights, we are using pre trained weights.
        # self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet_model(x)
        output = output.view(output.size()[0], -1)
        return output
    
    def forward_views(self, view_inputs):
        y = self.resnet_model(view_inputs)
        # print(y.shape)
        # y = y.view((int(view_inputs.shape[0]/self.n_views),self.n_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        y = y.view((int(view_inputs.shape[0]/self.n_views),self.n_views,y.shape[-3],-1))#(8,12,512,1)

        y = y.view((int(view_inputs.shape[0]/self.n_views), self.n_views * self.fc_in_features))

        # This was done for mvcnn, but it was pretrained as a classifier so
        # return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))
        view_comb = self.view_comb(y)

        return view_comb

    def forward(self, view_inputs, query_image):
        # get two images' features
        query_feats = self.forward_once(query_image)
        # print(query_feats.shape)
        view_feats = self.forward_views(view_inputs)
        # print(view_feats.shape)

        # concatenate both images' features
        output = torch.cat((view_feats, query_feats), 1)
        # pass the concatenation to the linear layers
        output = self.fc(output)
        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
    
    
def train_test_category(category:str, train_model=True, load_model=False):
    batch_size = 8
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    epochs = 10
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v1_occluded = '/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded'


    ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1_occluded, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)
    test_ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1_occluded, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)

    # Creating data indices for training and validation splits:
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=test_sampler)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_comparison/', category + "_multiview_model_occ_finetune.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_comparison/', category + "_multiview_log_occ_finetune.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}


def train_test_difference_category(category:str, train_model=True, load_model=False):
    batch_size = 8
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    epochs = 5
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1 2'

    ds = ViewCombDifferenceDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)          
    test_ds = ViewCombDifferenceDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)

    # Creating data indices for training and validation splits:
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=test_sampler)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_difference/', category + "_multiview_model.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_difference/', category + "_multiview_log.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}


def train_test_unseen_model(category:str, train_model=True, load_model=False):
    batch_size = 8
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    epochs = 2
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded'
    
    ds = ViewCombUnseenModelDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                                    n_views=12, n_samples=12, transforms=preprocess, data_split='train', train_split=1, seed=seed)
    val_ds = ViewCombUnseenModelDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                                    n_views=12, n_samples=12, transforms=preprocess, data_split='val', train_split=1, seed=seed)
    test_ds = ViewCombUnseenModelDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                                    n_views=12, n_samples=12, transforms=preprocess, data_split='test', train_split=1, seed=seed)

    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)

    # Would a scheduler be benificial here?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_unseen/', category + "_multiview_model.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_unseen/', category + "_multiview_log.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}

    
def train_test_category_v2(category:str, train_model=True, load_model=False):
    batch_size = 8
    epochs = 10
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v2 = '/Users/bprovan/University/dissertation/datasets/images_ds_v2'
    
    ds = ViewCombDatasetv2(missing_parts_base_dir_v2, category, transforms=preprocess, split='train', seed=seed)
    val_ds = ViewCombDatasetv2(missing_parts_base_dir_v2, category, transforms=preprocess, split='validation', seed=seed)
    test_ds = ViewCombDatasetv2(missing_parts_base_dir_v2, category, transforms=preprocess, split='test', seed=seed)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_comparison/', category + "_multiview_model_2.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_comparison/', category + "_multiview_log_2.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}

    
def train_test_category_unseen_anomaly(category:str, train_model=True, load_model=False):
    batch_size = 8
    epochs = 10
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v2 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded_unseen_anomaly2'
    
    ds = ViewCombDatasetUnseenAnomaly(missing_parts_base_dir_v2, category, transforms=preprocess, split='train', seed=seed)
    val_ds = ViewCombDatasetUnseenAnomaly(missing_parts_base_dir_v2, category, transforms=preprocess, split='validation', seed=seed)
    test_ds = ViewCombDatasetUnseenAnomaly(missing_parts_base_dir_v2, category, transforms=preprocess, split='test', seed=seed)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_comparison/', category + "_multiview_model_2.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_comparison/', category + "_multiview_log_2.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc = evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

    return {category: {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}


def train_test_all_categories(categories, train_model=True, load_model=False):
    batch_size = 8
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    epochs = 10
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v1_occluded = '/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded'


    ds = ViewCombDatasetAllCats(img_dir=missing_parts_base_dir_v1_occluded, categories=categories, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)
    test_ds = ViewCombDatasetAllCats(img_dir=missing_parts_base_dir_v1_occluded, categories=categories, 
                         n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)

    # Creating data indices for training and validation splits:
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                        sampler=test_sampler)
    
    print(len(train_dataloader))
    print(len(val_dataloader))

    model = ViewCombNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_comparison/',  "_multiview_model_all.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_comparison/', "_multiview_log_all.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    # total_acc, test_loss, precision, class0_acc, class1_acc, all_res_dict = evaluate_multiview_all(model, device, test_dataloader, criterion, set="Test")
    # print(all_res_dict)

    if load_model:
        model = model_saver.load_model(model, optimizer)

    if train_model:
        for epoch in tqdm(range(1, epochs + 1)):
            train_multiview_epoch_all(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, 
                                  model_saver, metric_logger)
            
        metric_logger.save_metrics()

    total_acc, test_loss, precision, class0_acc, class1_acc, all_res_dict = evaluate_multiview_all(model, device, test_dataloader, criterion, set="Test")

    return {"overall": {"accuracy": total_acc,
                       "avg loss" : test_loss,
                       "precision": precision,
                       "class0_acc": class0_acc,
                       "class1_acc": class1_acc}}, all_res_dict


if __name__ == "__main__":
    # main()

    categories = [
        'KitchenPot', 
        'USB', 'Cart', 
        'Box', 
        'Pliers', 
        'WashingMachine', 
                'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot'
                ]

    # unseen anomaly
    # categories = ["Cart", "Eyeglasses", "Faucet", "Luggage", "Oven",
    #                 "Scissors", "Sitting Furniture", "Switch", "Table", "Toilet", "Trashcan", "Window"]
    
    # categories = ["Box", "KitchenPot"]

    # train_test_unseen_category(category)
    # train_test_difference_category(category)
    # train_test_category(category)
    # train_test_category_v2(category)
    # train_test_category_v2(category, train_model=False, load_model=True)

    
    overall_dict, all_res_dict = train_test_all_categories(categories, train_model=False, load_model=True)

    with open('logs/multiview_all_cats.json', 'w') as fp:
            json.dump(all_res_dict, fp)

    with open('logs/multiview_overllall_cats.json', 'w') as fp:
            json.dump(overall_dict, fp)

    # for category in categories:
    #     print(category)
    # #     # Train a model from scratch
    #     train_test_difference_category(category, train_model=True, load_model=False)
    #     # res_dict = train_test_category_unseen_anomaly(category, train_model=False, load_model=True)
        
    #     # res_dict = train_test_category_v2(category, train_model=False, load_model=True)
    #     all_res_dict.update(res_dict)
    #     print("FINISHED: ", category, "\n")

    #     with open('logs/multiview_unseen_anomaly_2.json', 'w') as fp:
    #         json.dump(all_res_dict, fp)

    