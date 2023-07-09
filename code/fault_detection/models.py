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


# https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
# Can also just create a more complex prediction 'head' model. eg a few layers with some relu


class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_size: int = 2, pretrained_imagenet: bool = True):
        super(ResNetEmbedding, self).__init__()

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
        # so then it returns feature vector of size 2048
        # This is the embedding network
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
        )

    def forward_once(self, x):
        output = self.resnet_model(x)
        output = output.view(output.size()[0], -1)
        return output


    def forward(self, x1):
        x1 = self.forward_once(x1)

        feats = self.fc(x1)

        return feats
    


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

