"""

This will form the baseline binary fault classification.
2 main models
- Most basic one which takes in one image as imput, and trys to detect whether it has a part missing or not (fault)
- base it off https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

try this first
https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch


- feature vector stacking
One that takes in multiple images, a feature vector is taken from each of them, and they are stacked together, 
and then a classifier is trained on them.

There is also yolov6 in pytorch, https://github.com/meituan/YOLOv6/blob/main/turtorial.ipynb
Could use that as another feature extractor (This time trained on COCO), so higher resolution. (COCO 640 x 336?) something like that

"""

import torch
import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np

# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")

device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

# for baseline, use pretrained on imagenet, binary classification. 

# Train, test, validation split
# For initial baseline train test split, we can just randomly split up the images from the file paths.

# so our initial size of photo is 1920x1080, we probably need to change this at some point

# for imagenet it is 284??
input_size = 284

data_dir = '/Users/bprovan/Desktop/glasses_basic/'


# Other augentation techniques??
trainTansform = transforms.Compose([
    transforms.CenterCrop(1080),
    transforms.Resize(input_size),
	# transforms.RandomResizedCrop(config.IMAGE_SIZE),
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	# transforms.Normalize(mean=config.MEAN, std=config.STD)
])
valTransform = transforms.Compose([
    transforms.CenterCrop(1080),
    transforms.Resize(input_size),
	transforms.ToTensor(),
	# transforms.Normalize(mean=config.MEAN, std=config.STD)
])

ds = datasets.ImageFolder(root=data_dir, transform=trainTansform)

batch_size = 32
validation_split = 0.2
test_split = 0.2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(ds)
indices = list(range(dataset_size))
val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
test_split_index = int(np.floor(dataset_size * (1 - (test_split))))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[val_split_index:test_split_index], indices[test_split_index:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler()

# Should work for the basic train test split
train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                           sampler=train_sampler)
val_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                sampler=val_sampler)
test_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                sampler=test_sampler)