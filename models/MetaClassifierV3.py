from torch import nn
import torch.nn.functional as F
from typing import Callable
import torch
from models import BaseModel
from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from models.Inception_V3 import Inception_V3
from models.densenet161 import Densenet_161
from models.densenet201 import Densenet_201
from models.vgg16 import VGG_16
from torchvision.transforms import Resize

class MetaClassifierV3(BaseModel):
    """
    A meta classifier using trained models to give base probabilities which then get used
    in a few linear layers to form a final adjusted probability
    """
    def __init__(self, num_classes: int = 2, loss_fn: Callable = F.cross_entropy, print_metrics: bool = True) -> None:
        super(MetaClassifierV3, self).__init__(num_classes, loss_fn, print_metrics)

        # Resize Transforms Required
        self.resize224 = Resize((224, 224))
        self.resize299 = Resize((299, 299))

        # Models used for meta classifier
        self.resnet34 = ResNet34.load_from_checkpoint(r"saved/ResNet34/Run_ID__2023-02-09__14-36-03/checkpoints/epoch=87-val_loss=0.69-val_acc=0.78.ckpt")
        self.inceptionv3 = Inception_V3.load_from_checkpoint(r"saved/Inception_V3/Run_ID__2023-02-09__15-38-08/checkpoints/epoch=37-val_loss=0.78-val_acc=0.78.ckpt")
        self.densenet161_1 = Densenet_161.load_from_checkpoint(r"saved/Densenet_161/Run_ID__2023-02-09__16-24-45/checkpoints/epoch=28-val_loss=0.45-val_acc=0.82.ckpt")
        self.densenet161_2 = Densenet_161.load_from_checkpoint(r"saved/Densenet_161/Run_ID__2023-02-09__16-01-11/checkpoints/epoch=80-val_loss=0.56-val_acc=0.82.ckpt")
        self.vgg16 = VGG_16.load_from_checkpoint(r"saved/VGG_16/Run_ID__2023-02-09__17-17-05/checkpoints/epoch=51-val_loss=0.60-val_acc=0.75.ckpt")
        self.resnet50 = ResNet50.load_from_checkpoint(r"saved/ResNet50/Run_ID__2023-02-13__15-45-23/checkpoints/epoch=30-val_loss=0.60-val_acc=0.78.ckpt")
        self.densenet201 = Densenet_201.load_from_checkpoint(r"saved/Densenet_201/Run_ID__2023-02-14__20-42-49/checkpoints/epoch=60-val_loss=0.82-val_acc=0.79.ckpt")

        # Freeze models
        for p in self.resnet34.parameters():
            p.requires_grad = False
        for p in self.inceptionv3.parameters():
            p.requires_grad = False
        for p in self.densenet161_1.parameters():
            p.requires_grad = False
        for p in self.densenet161_2.parameters():
            p.requires_grad = False
        for p in self.vgg16.parameters():
            p.requires_grad = False
        for p in self.resnet50.parameters():
            p.requires_grad = False
        for p in self.densenet201.parameters():
            p.requires_grad = False

        # Linear layers
        self.fc1 = nn.Linear(14, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 80)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 48)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(48, 24)
        self.dropout4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(24, 2)

        # Utility layers
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Resize x to different sizes
        x_224 = self.resize224(x)
        x_299 = self.resize299(x)

        # Pass x into each model and take softmax
        resnet34_out = self.resnet34(x_224)
        resnet34_out = self.sm(resnet34_out)

        inceptionv3_out = self.inceptionv3(x_299)
        inceptionv3_out = self.sm(inceptionv3_out)

        densenet161_1_out = self.densenet161_1(x_224)
        densenet161_1_out = self.sm(densenet161_1_out)

        densenet161_2_out = self.densenet161_2(x_224)
        densenet161_2_out = self.sm(densenet161_2_out)

        vgg16_out = self.vgg16(x_224)
        vgg16_out = self.sm(vgg16_out)

        resnet50_out = self.resnet50(x_224)
        resnet50_out = self.sm(resnet50_out)

        densenet201_out = self.densenet201(x_224)
        densenet201_out = self.sm(densenet201_out)

        # Combine outputs by concatenation
        out = torch.cat([resnet34_out, inceptionv3_out, densenet161_1_out, densenet161_2_out, vgg16_out, resnet50_out, densenet201_out], dim=1)

        # Pass outputs through linear layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc5(out)
        
        return out