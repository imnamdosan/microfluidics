# %% Imports
from models.resnet34 import ResNet34
from models.MetaClassifierV2 import MetaClassifierV2
from dataModules.MfdDataModule import MfdDataModule, gray_loader
from torchvision.transforms import ToTensor, Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, Compose
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
import numpy as np

# %% Datamodule
for s in range(224, 225):
    torch.manual_seed(42)
    np.random.seed(42)
    train_transforms = [ToTensor(), RandomHorizontalFlip(), RandomVerticalFlip(), Normalize([0.1553], [0.1838])]
    val_transforms = [ToTensor(), Normalize([0.1553], [0.1838])]
    dm = MfdDataModule("data/364/", 1, 4, 0.8, True, train_transforms, val_transforms)
    dm.setup()

    ds = ImageFolder(root="data/364/", loader=gray_loader, transform=Compose(val_transforms))

    num_classes = 2
    acc = MulticlassAccuracy(num_classes, average="weighted")
    cm = MulticlassConfusionMatrix(num_classes)

    # % Restore model
    model = MetaClassifierV2.load_from_checkpoint(r"saved/MetaClassifierV2/Run_ID__2023-02-16__01-51-52/checkpoints/epoch=59-val_loss=1.71-val_acc=0.88.ckpt")
    # model = Inception_V3.load_from_checkpoint(r"epoch=42-val_loss=0.34-val_acc=0.88.ckpt")
    model.eval()

    data = dm.val
    data_indices = dm.valid_indices
    mislabelled = []
    mislabelled_preds = []
    for i in range(len(data)):
        idx_of_image = data_indices[i]
        input, label = data[i]
        input = torch.unsqueeze(input, 0)
        label = torch.tensor([label])
        pred = model(input)

        acc.update(pred, label)
        cm.update(pred, label)

        if torch.argmax(pred, dim=1) != label:
            mislabelled.append(idx_of_image)
            mislabelled_preds.append(pred)

    accuracy = acc.compute()
    conf_matrix = cm.compute()
    print(s)
    print(f"{accuracy:5.2%}")
    print(conf_matrix)

# %% View mislabelled images
idx = 2
img_idx = mislabelled[idx]
ds = ImageFolder("data/364/", loader=gray_loader)
img = ds[img_idx][0]
plt.imshow(img, cmap="gray")
print("Label is " + str(ds[img_idx][1]))
pred = mislabelled_preds[idx]
print("Predicted was " + str(pred) + " which is " + str(torch.squeeze(torch.argmax(pred, dim=1)).item()))
# %%
