from torch import nn
import torch.nn.functional as F
from typing import Callable
import torch
from models import BaseModel
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception_V3(BaseModel):
    """
    INCEPTION net or also called GoogleNetv3, a famous convNEt trained on Imagenet from 2015
    """
    def __init__(self, num_classes: int = 2, loss_fn: Callable = F.cross_entropy, print_metrics: bool = True) -> None:
        super(Inception_V3, self).__init__(num_classes, loss_fn, print_metrics)

        # Required layers for model
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.aux_logits = False
        # Readjust last layer to be same number of classes as ours
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        # X is (batch, 1, 299, 299)
        x = torch.cat((x,x,x),dim=1)
        # X is (batch, 3, 299, 299)
        x = self.model(x)
        
        return x