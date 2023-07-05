# Try to use the triplet loss, see if this helps at all. The question becomes how to check whether they are different or not?
# No longer pushing them towards 0 or 1...

# Apply it to cats and dogs, and to eyeglasses?
# Code adapted from https://github.com/adambielski/siamese-triplet/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm


from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import TripletDatasetCatsDogs

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

class TripletNetwork(nn.Module):
    def __init__(self, embedding_size: int = 2):
        super(TripletNetwork, self).__init__()
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

        self.fc_in_features = self.resnet_model.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

        # add linear layers to compare between the features of the two images
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_in_features * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        # )

        # This is the embedding network
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
        )

        # initialize the weights, we are using pre trained weights.

    def forward_once(self, x):
        output = self.resnet_model(x)
        output = output.view(output.size()[0], -1)
        return output


    def forward(self, x1, x2, x3):
        x1_res = self.forward_once(x1)
        x2_res = self.forward_once(x2)
        x3_res = self.forward_once(x3)

        feats_1 = self.fc(x1_res)
        feats_2 = self.fc(x2_res)
        feats_3 = self.fc(x3_res)

        return feats_1, feats_2, feats_3
    

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
  

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # What about this compare to the other losses? 
    # there is triplet margin loss with distance too, can input the distance to use.
    criterion = nn.TripletMarginLoss()

    # I suppose you can have a two step training process, eg do the triplet loss first,
    #  and then after (for similarity) use sigmoid for being close or not... This would use BCE then...

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        outputs = model(anchor, positive, negative)
        loss = criterion(*outputs)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
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

    # ds = SiameseDatasetCatsDogs(img_dir=cats_dogs_data_dir, transforms=preprocess)
    # ds = SiameseDatasetSingleCategory(img_dir=missing_parts_base_dir, category="KitchenPot", transforms=preprocess)
    ds = TripletDatasetCatsDogs(img_dir=cats_dogs_data_dir, transforms=preprocess)


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

    model = TripletNetwork(embedding_size=2).to(device)
    # Maybe change to Adam?
    optimizer = optim.Adam(model.parameters())

    # scheduler = StepLR(optimizer, step_size=1)
    for epoch in tqdm(range(1, epochs + 1)):
        train(model, device, train_dataloader, optimizer, epoch)
        # test(model, device, train_dataloader, test_loader_len=len(train_indices))
        # test(model, device, val_dataloader, test_loader_len=len(val_indices))
        # scheduler.step()

if __name__ == "__main__":
    main()

