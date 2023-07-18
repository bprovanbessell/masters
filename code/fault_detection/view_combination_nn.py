# Architecture like a Siamese NN
# Have a feature extraction network, Generate 12 feature sets for the comparison -> Hakan said to average pool them (how exactly??) 
# That doesnt work over feature maps...

# Or take some architecture from the 3D classification papers!?

# INITIAL BASELINE, DO AS IN SIAMESE NN: CONCATENATE FEATURES, USE LINEAR NETWORK TO COMPARE THEM ALL!

# MVCNN VIEW POOLING

# ROTATIONNET TYPE VIEW COMBINATIONS

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from custom_dataset import ViewCombDataset, ViewCombDifferenceDataset
from trainer import train_multiview_epoch
from eval import evaluate_multiview, ModelSaver
from logger import MetricLogger
import os

class ViewCombNetwork(nn.Module):
    def __init__(self, n_views=12):
        super(ViewCombNetwork, self).__init__()

        self.n_views = n_views
        # get resnet model
        # Try with pretrained and non pretraind
        # change that here...
        # Try a more basic resnet18, cheaper
        # weights = ResNet50_Weights.IMAGENET1K_V2
        weights = ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        self.resnet_model = resnet18(weights=weights)
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze the weights, set them to be non trainable
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        self.fc_in_features = self.resnet_model.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

        # We want to reduce this to the number of features we have
        self.view_comb = nn.Sequential(
            nn.Linear(self.fc_in_features * self.n_views, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.fc_in_features),
        )

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
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
    

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, ((view_images, query_image), targets) in enumerate(train_loader):

        N,V,C,H,W = view_images.size()
        # print(N,V,C,H,W)
        view_images = view_images.view(-1,C,H,W).to(device)
        # print(view_images.shape)

        # print(query_image.shape)

        view_images, query_image, targets = view_images.to(device), query_image.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(view_images, query_image).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(query_image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # if args.dry_run:
            #     break

def test(model, device, test_loader, test_loader_len):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for ((view_images, query_images), targets) in test_loader:
            N,V,C,H,W = view_images.size()
            # print(N,V,C,H,W)
            view_images = view_images.view(-1,C,H,W).to(device)
            view_images, query_images, targets = view_images.to(device), query_images.to(device), targets.to(device)
            outputs = model(view_images, query_images).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            # print("pred", pred)
            # print("targets", targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= test_loader_len

    # With cats and dogs, can achieve 95% accuracy in the first epoch.
    # But for some reason, does not work with the Kitchen Pot category.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * correct / test_loader_len))
    
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

    category = "KitchenPot"
    ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=True, seed=seed)
    test_ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1, category=category, 
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
                                    sampler=train_sampler)
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

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/multiview_comparison/', category + "_multiview_model.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/multiview_comparison/', category + "_multiview_log.json")

    model_saver = ModelSaver(model_save_path)
    metric_logger = MetricLogger(metric_save_path)

    for epoch in tqdm(range(1, epochs + 1)):
        train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch, model_saver, metric_logger)

    evaluate_multiview(model, device, test_dataloader, criterion, set="Test")
    

def main():

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

    # ds = SiameseDatasetCatsDogs(img_dir=cats_dogs_data_dir, transforms=preprocess)
    # ds = SiameseDatasetSingleCategory(img_dir=missing_parts_base_dir, category="KitchenPot", transforms=preprocess)
    # ds = SiameseDatasetPerObject(img_dir=missing_parts_base_dir, category="EyeGlasses", n=8, transforms=preprocess, train=False)
    category = "KitchenPot"
    ds = ViewCombDifferenceDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                         n_views=12, n_samples=12, transforms=preprocess, train=True, seed=seed)
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
                                    sampler=train_sampler)
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

    model_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/models/', category + "_multiview_model.pt")
    metric_save_path = os.path.join('/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/', category + "_multiview_log.json")

    model_saver = ModelSaver(model_save_path)

    metric_logger = MetricLogger(metric_save_path)

    # scheduler = StepLR(optimizer, step_size=1)
    for epoch in tqdm(range(1, epochs + 1)):
        # train(model, device, train_dataloader, optimizer, epoch)
        train_multiview_epoch(model, device, train_dataloader, val_dataloader, optimizer, criterion, epoch)

        # test(model, device, train_dataloader, test_loader_len=len(train_indices))
        # test(model, device, val_dataloader, test_loader_len=len(val_indices))
        # scheduler.step()
    # test(model, device, test_dataloader, test_loader_len=len(test_indices))
    evaluate_multiview(model, device, test_dataloader, criterion, set="Test")

if __name__ == "__main__":
    # main()

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']
    
    category = "KitchenPot"
    
    train_test_category(category, train_model=True, load_model=False)
    