# Siamese NN to perform image similarity
# Take the input of two images, and then do a comparison between them. 
# The network is composed of two identical networks, one for each input.
# The output of each network is concatenated and passed to a linear layer. 
# The output of the linear layer passed through a sigmoid function.
# `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
# This implementation varies from FaceNet as we use the `ResNet-18` model from
# `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
# In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
# based on https://github.com/pytorch/examples/blob/main/siamese_network/main.py

import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
import glob
from tqdm import tqdm

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
from custom_dataset import SiameseDatasetSingleCategory


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        # Try with pretrained and non pretraind
        # change that here...
        weights = ResNet50_Weights.IMAGENET1K_V2
        # weights = ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        self.resnet_model = resnet50(weights=weights)
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze the weights, set them to be non trainable
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet_model.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

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


    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
    
class SiameseDatasetCatsDogs(Dataset):

    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir

        self.transforms = transforms

        self.imgs_paths = glob.glob(img_dir + '/*/*.jpg')
        # get the labels of the images paths, this is needed for the selection stage
        self.groups = {0:[],
                       1:[]}

        # We just have positive and negative images
        for img_path in self.imgs_paths:
            label_str = img_path.split('/')[-1].split('.')[0]

            # for now just binary classification
            if label_str == 'cat':
                # label = torch.zeros((1), dtype=torch.float32)
                self.groups[0].append(img_path)
            else:
                self.groups[1].append(img_path)


    def __getitem__(self, idx):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """
        selected_class = random.randint(0, 1)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.groups[selected_class]) -1)
        
        # pick the index to get the first image
        img_path_1 = self.groups[selected_class][random_index_1]

        # get the first image
        img_1 = Image.open(img_path_1).convert("RGB")


        # same class
        if idx % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.groups[selected_class]) -1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.groups[selected_class]) -1)
            
            # pick the index to get the second image
            img_path_2 = self.groups[selected_class][random_index_2]

            # get the second image
            img_2 = Image.open(img_path_2).convert("RGB")

            # set the label for this example to be positive (1), similarity of 1
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 1)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 1)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.groups[other_selected_class])-1)

            # pick the index to get the second image
            img_path_2 = self.groups[other_selected_class][random_index_2]

            # get the second image
            img_2 = Image.open(img_path_2).convert("RGB")

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        if self.transforms is not None:
            img_1 = self.transforms(img_1)
            img_2 = self.transforms(img_2)

        return img_1, img_2, target

    
    def __len__(self):
        return len(self.imgs_paths)
    

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
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
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            print("pred", pred)
            print("targets", targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= test_loader_len

    # With cats and dogs, can achieve 95% accuracy in the first epoch.
    # But for some reason, does not work with the Kitchen Pot category.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * correct / test_loader_len))
    

def main():

    batch_size = 64
    validation_split = 0.2
    test_split = 0
    shuffle_dataset = True
    random_seed= 42
    epochs = 20

    weights = ResNet50_Weights.IMAGENET1K_V2
    # weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()


    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    cats_dogs_data_dir = '/Users/bprovan/University/dissertation/masters/code/data/archive/train'
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'

    ds = SiameseDatasetCatsDogs(img_dir=cats_dogs_data_dir, transforms=preprocess)
    ds = SiameseDatasetSingleCategory(img_dir=missing_parts_base_dir, category="KitchenPot", transforms=preprocess)


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
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        sampler=test_sampler)

    model = SiameseNetwork().to(device)
    # Maybe change to Adam?
    optimizer = optim.Adadelta(model.parameters())

    scheduler = StepLR(optimizer, step_size=1)
    for epoch in tqdm(range(1, epochs + 1)):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, train_dataloader, test_loader_len=len(train_indices))
        test(model, device, val_dataloader, test_loader_len=len(val_indices))
        scheduler.step()

if __name__ == "__main__":
    main()

