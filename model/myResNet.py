from torch import nn
import torch
import torchvision

# class myResNet(nn.Module):
    
#     def __init__(self, num_classes):
#         super(myResNet, self).__init__()
#         num_feats = self.fc.in_features
#         self.resnet = torchvision.models.resnet18(num_classes=num_feats)
#         self.resnet.fc = nn.Linear(num_feats, num_classes)
        
#     def forward(self, x):
#         x = self.resnet(x)
        
#         return x
    
def myResNet(num_classes):
    model = torchvision.models.resnet18()
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, num_classes)
    
    return model
