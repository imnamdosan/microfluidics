import pytorch_lightning as pl
import torch
import cv2
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as t
from PIL import Image
from typing import List, Union
from numpy.random import shuffle
import numpy as np
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
import cv2


class MfdDataModule(pl.LightningDataModule):
    """
    Encapsulates the dataset to allow the Trainer to retrieve the appropriate dataloaders
    and apply relevant transforms.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, train_percent: float, balanced: bool = False, train_transforms: Union[List, None] = None, val_transforms: Union[List, None] = None) -> None:
        """
        Sets up desired hyperparameters and directory to retrieve dataset from 

        - data_dir: str - Directory where dataset is downloaded in and where to find data
        - batch_size: int - The desired loading batch size for dataloaders
        - num_workers: int - The number of workers given to each dataloader
        - train_percent: float - The ratio used to split the dataset into the training set 
        - balanced: bool - A flag for whether to use a balanced dataloader or not via weighted resampling
        - train_transforms: list - An array of all transforms specified in the config file for training dataset
        - val_transforms: list - An array of all transforms specified in the config file for validation dataset
        """

        super().__init__()

        # Save hyperparameters 
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.num_workers = num_workers
        if train_transforms is not None:
            self.train_transforms = t.Compose(train_transforms)
        else:
            self.train_transforms = None
        if val_transforms is not None:
            self.val_transforms = t.Compose(val_transforms)
        else:
            self.val_transforms = None
        self.train_percent = train_percent
        self.balanced = balanced

    def setup(self, stage=None) -> None:
        """
        Obtains the dataset using the ImageFolder function and splits it into training and validation sets 
        based on the specified ratio 
        """

        # Instantiate dataset by retrieving the images from the dataset directory 
        # ImageFolder can be used because folder hierarchy is split: /data/Abnormal & /data/Normal
        self.train_data = ImageFolder(self.data_dir, loader=SMA, transform=self.train_transforms)
        self.val_data = ImageFolder(self.data_dir, loader=SMA, transform=self.val_transforms)

        # Split dataset into training and validation sets 
        ds_length = len(self.train_data)
        indices = list(range(ds_length))
        split_idx = int(self.train_percent*ds_length)
        shuffle(indices)
        self.train_indices, self.valid_indices = indices[:split_idx], indices[split_idx:]
        self.train = Subset(self.train_data, self.train_indices)
        self.val = Subset(self.val_data, self.valid_indices)
        self.num_classes = len(unique([i[1] for i in self.train.dataset]))  

    def train_dataloader(self):
        # If not balanced, use regular dataloader
        if not self.balanced:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        # For unbalanced dataset we create a weighted sampler
        weights = make_weights_for_balanced_classes(self.train, self.num_classes)
        weights = torch.DoubleTensor(weights)

        # Set replacement=True for oversampling the minority class
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_dl = DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, pin_memory=True, drop_last=True)
        # Visualize DataLoader Distribution
        # class_0_batch_counts, class_1_batch_counts, idx_seen  = visualise_dataloader(train_dl, {0: "Abnormal(Majority Class)", 1: "Normal(Minority Class)"})
        return train_dl

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# def gray_loader(path: str) -> Image:
#     """
#     Custom loader to prevent ImageFolder converting images to RGB automatically.

#     Args:
#     - path: str
#         - The path of the image to be loaded.
#     """
#     # Open TIFF 16 bit image as np array
#     img = np.asarray(Image.open(path))  
    
#     # Force values to be within 0-1 (65535 due to 16 bit image)
#     img = img / 65535
#     return img

def SMA(path: str) -> Image:
    """
    Args: 
    - path: str
        - The path of the image to be loaded

    Function used to denoise and create mask within the image using steps below

    DENOISING:

    1. Y-channel Sharpening(applying convolution filters, get the y chanel by indexing)

    2. Gaussian Filter (Smoothen out the image)

    Mask Creation:
    
    1. Sobel Edge Detector(This creates the mask that we want)

    2. Remove small objects still present in the image using thresholding methods

    3. Apply Morphological Erosion to reduce size of noise  
    
    4. Apply Median Filter 

    This function will be used as our loader in ImageFolder

    """
     # Open TIFF 16 bit image as np array
    img = np.asarray(Image.open(path))  
    
    # Force values to be within 0-1 (65535 due to 16 bit image)
    img = img / 65535

    # DENOISING
    # Using a mexican hat or laplacian filter to 
    filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    # Applying the filter on our images
    img = cv2.filter2D(img, -1, filter)
    # Applying gaussian blur on the images using a gaussian filter with of kernel size 5
    img = cv2.GaussianBlur(img,(5,5),0)
    
    # Mask Creation
    # Apply sobel detector to obtain edges in the image
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Calculating Magnitude as the Image
    img = np.sqrt(np.square(sobelX) + np.square(sobelY))
    # Apply thresholding to the image
    threshold_value = np.mean(img)
    thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    # Apply morphological operations to remove small objects
    kernel = np.ones((3,3))
    img = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    img = medfilt2d(img, kernel_size=3)
    return img

# Function to find unique values within a list
def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    
    return unique_list

def make_weights_for_balanced_classes(images, nclasses):
    """
    Returns weights for WeightedRandomSampler.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                         
    for i in range(nclasses):                                          
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

# To visualize distribution of data in each classes for each batch
def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []

    for i, batch in enumerate(dl):

        idxs = batch[0][:, 0].tolist()
        classes = batch[1][:]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            raise ValueError("More than two classes detected")

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 15))

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            label=(id_to_label[0] if id_to_label is not None else "0"),
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.set_xticks(ind, ind + 1)
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        plt.show()

        num_images_seen = len(idxs_seen)

        print(
            f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
        )
        print("=============")
        print(f"Num. unique images seen: {len(unique(idxs_seen))}/{total_num_images}")
    return class_0_batch_counts, class_1_batch_counts, idxs_seen

# Function to find unique values within a list
def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    
    return unique_list
def sharpen(YCbCr_images):#Sharpens the images to enhance edges of the Y channel of images

    #Splitting the Y, Cb, Cr from the YCbCr images 
    Y, Cr, Cb = cv2.split(YCbCr_images)

    #Using a mexican hat or laplacian filter to 
    filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])

    #Applying the filter on our images
    sharp_img = cv2.filter2D(Y, -1, filter)

    return sharp_img

def blur(images):
    
    #applying gaussian blur on the sharp images using a gaussian filter with of kernel size 5
    blur = cv2.GaussianBlur(images,(5,5),0)

    return blur

