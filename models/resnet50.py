from torch import nn
import torch.nn.functional as F
from typing import Callable
import torch
from models import BaseModel
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(BaseModel):
    """
    A Resnet 50 network pretrained on Image Net
    """
    def __init__(self, num_classes: int = 2, loss_fn: Callable = F.cross_entropy, print_metrics: bool = True) -> None:
        super(ResNet50, self).__init__(num_classes, loss_fn, print_metrics)

        # Required layers for model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Readjust last layer to be same number of classes as ours
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        # X is (batch, 1, 224, 224)
        x = torch.cat((x,x,x),dim=1)

        # X is (batch, 3, 224, 224)
        x = self.model(x)
        return x