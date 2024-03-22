import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from models import BaseModel
from typing import Callable
import torch.nn.functional as F
        
class Conv(nn.Module): #Notebook 4
    """
    Creates and passes a convolutional layer, batch normalisation layer and leaky ReLU activation 
    function as a singular block rather than three individual layers for ease of implementation 
    into model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0):
        super().__init__()
        # Required layers for the block
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.01, inplace = False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class Residual(nn.Module):
    """
    Creates and passes two convolutional layers as a singular block rather
    than two individual layers for ease of implementation into model.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Required layers for the block
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1,  padding = 0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3,  padding = 1)
        
        #Adding
    def forward(self, x):
        #Save original
        identity = x 

        #Pass through layers
        out = self.conv1(x)
        out = self.conv2(out)

        # Apply skip connection
        out += identity

        return out

class CCR(nn.Module):
    """
    Creates and passes two convolutional blocks and a residual block as a singular block
    rather than three individual blocks for ease of implementation into model. 
    """
    def __init__(self, conv1_in, conv1_out, conv2_out):
        super().__init__()
        # Required layers for the block
        self.conv1 = Conv(conv1_in, conv1_out, kernel_size = 1, padding = 0)
        self.conv2 = Conv(conv1_out, conv2_out, kernel_size = 3, padding = 1)
        self.residual = Residual(conv2_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        return x


"""
Left to do:
Debug and train
"""   


class YOLOv3(BaseModel):
    """
    YOLOv3 model that uses Darknet53 architecture for object detection.
    """
    def __init__(self, num_classes: int = 2, loss_fn: Callable = nn.CrossEntropyLoss, print_metrics: bool = True) -> None:
        super(YOLOv3, self).__init__(num_classes, loss_fn, print_metrics)
        
        # Required layers for the model
        self.conv1 = Conv(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv2 = Conv(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.ccr1 = CCR(conv1_in=64,conv1_out=32,conv2_out=64)
        self.conv3 = Conv(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.ccr2_1 = CCR(conv1_in=128,conv1_out=64,conv2_out=128)
        self.ccr2_2 = CCR(conv1_in=128,conv1_out=64,conv2_out=128)
        self.conv4 = Conv(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.ccr3_1 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_2 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_3 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_4 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_5 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_6 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_7 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.ccr3_8 = CCR(conv1_in=256,conv1_out=128,conv2_out=256)
        self.conv5 = Conv(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)
        self.ccr4_1 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_2 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_3 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_4 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_5 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_6 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_7 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.ccr4_8 = CCR(conv1_in=512,conv1_out=256,conv2_out=512)
        self.conv6 = Conv(in_channels=512,out_channels=1024,kernel_size=3,stride=2,padding=1)
        self.ccr5_1 = CCR(conv1_in=1024,conv1_out=512,conv2_out=1024)
        self.ccr5_2 = CCR(conv1_in=1024,conv1_out=512,conv2_out=1024)
        self.ccr5_3 = CCR(conv1_in=1024,conv1_out=512,conv2_out=1024)
        self.ccr5_4 = CCR(conv1_in=1024,conv1_out=512,conv2_out=1024)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(in_features=1024,out_features=1000)
        self.linear2 = nn.Linear(in_features=1000,out_features=num_classes)
        
        #DONEnn.adaptive avgpool - set to 1
        #DONEuse nn.flatten dim=1
        #DONE2x nn.linear - 1024 to 1000, 1000 to 2

    def forward(self, x):
        #Passing through layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ccr1(x)
        x = self.conv3(x)
        x = self.ccr2_1(x)
        x = self.ccr2_2(x)
        x = self.conv4(x)
        x = self.ccr3_1(x)
        x = self.ccr3_2(x)
        x = self.ccr3_3(x)
        x = self.ccr3_4(x)
        x = self.ccr3_5(x)
        x = self.ccr3_6(x)
        x = self.ccr3_7(x)
        x = self.ccr3_8(x)
        x = self.conv5(x)
        x = self.ccr4_1(x)
        x = self.ccr4_2(x)
        x = self.ccr4_3(x)
        x = self.ccr4_4(x)
        x = self.ccr4_5(x)
        x = self.ccr4_6(x)
        x = self.ccr4_7(x)
        x = self.ccr4_8(x)
        x = self.conv6(x)
        x = self.ccr5_1(x)
        x = self.ccr5_2(x)
        x = self.ccr5_3(x)
        x = self.ccr5_4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x