import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
from models import BaseModel


class modified_vgg16(BaseModel):
    """
    A VGG16 network modified to use a SELU activation function along with structure changes.
    """
    def __init__(self, learning_rate: float = 0.001, num_classes: int = 2, loss_fn: Callable = nn.CrossEntropyLoss, print_metrics: bool = True) -> None:
        super(modified_vgg16, self).__init__(num_classes, loss_fn, print_metrics)

        # Required layers for model

        # First block of layer for ease of implementation
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AlphaDropout(p=0.2)
        )

        # Second block of layer for ease of implementation
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AlphaDropout(p=0.2)
        )

        # Third block of layer for ease of implementation
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AlphaDropout(p=0.2)
        )

        # Fully connected block for ease of implementation
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=16384, out_features=1024),
            nn.SELU(),
            nn.Linear(1024, 1024),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        # x.shape: [batch_size, 1, 128, 128] 
        x = torch.cat((x,x,x),dim=1)
        # print("Input Shape: ", x.shape)
        # x.shape: [batch_size, 3, 128, 128] 
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        logits = self.classifier(x)
        # Logits provide an unnormalized score for each classes in an image
        # pred = torch.sigmoid(logits)
        # Sigmoid is then used to get the normalised score between 0 and 1
        return logits

