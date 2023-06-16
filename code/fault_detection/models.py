import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


def resnet50_pretrained_model(num_classes=1):

    weights = ResNet50_Weights.IMAGENET1K_V2
    # weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # freeze the weights, set them to be non trainable
    for param in model.parameters():
        param.requires_grad = False

    modelOutputFeats = model.fc.in_features

    model.fc = nn.Linear(modelOutputFeats, num_classes)

    return model

