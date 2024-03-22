# %%
import sys
sys.path.insert(0, r"path to your local git repo")
from dataModules import MfdDataModule
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import pywt

# %%
# Data module needs to be instantiated (edit the data dir to your own)
# Then run setup() as this creates our two datasets for training and validation.
# This process could also potentially be where the error is so please look into that as well.
dm = MfdDataModule(data_dir="data/", 
                    batch_size=10, 
                    num_workers=0, 
                    train_percent=0.8, 
                    train_transforms=[ToTensor()],
                    val_transforms=[ToTensor()])
dm.setup()


# Below is code to access the training dataset (keep in mind it is a "Subset", Im not 100% sure on how
# they work but pretty much identical to Dataset, it inherits from it)
train_set = dm.train

# Access the first image with its label in training dataset
first_img = dm.train[0]

# Access first image alone (image will have appropriate transforms given to it)
first_img = dm.train[0][0]

random.seed(32)
def get_image():
    # Access Nth image alone
    N = random.randint(0,20)
    N_img = dm.train[N][0]

    # View Nth image if the ToTensor() transform was given
    img = N_img.permute(1, 2, 0)
    img = np.array(img)
    
    return img

img1 = get_image()
img2 = get_image()
img3 = get_image()
img4 = get_image()
img5 = get_image()
img6 = get_image()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))

# Set the title for the subplot
fig.suptitle("Sobel Edge Detector, Thresholding and Median Filter")

# Add each image to the subplot
axes[0, 0].imshow(img1, cmap="gray")
axes[0, 1].imshow(img2, cmap="gray")
axes[0, 2].imshow(img3, cmap="gray")
axes[1, 0].imshow(img4, cmap="gray")
axes[1, 1].imshow(img5, cmap="gray")
axes[1, 2].imshow(img6, cmap="gray")

# Remove the axis ticks and labels for a cleaner look
for ax in axes.flatten():
    ax.axis("off")

# Display the subplot
plt.show()

# plt.imshow(img3, cmap="gray")     # permute is required since matplotlib expects Height x Width x Channels but
# #                                                     # Tensors are Channels x Height x Width


# plt.show()


