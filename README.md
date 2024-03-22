# MDN Microfluidics Project
**Last updated:** 15/03/2023
## What is this project?
This project aims to develop a deep learning classification model on spermatozoa morphology. It is conducted by the student engineering group, **Monash Deep Neuron**, with supervision from Dr. Reza Nosrati and PhD candidate, Ms. Sahar Shahali.

The members of the group are:
- Antonio Christophorus (Project Manager)
- Mubasshir Murshed
- Pranav Kutty
- Advay Kumar
- Satya Jhaveri
- Kabir Chugh
- David Ji


## What does this repository contain?
This repository contains the collection of files and models used. The dataset being used can be found in the *data* directory.

The environment used to run the repository can be found in ***environment.yml***.

All models developed are contained in the *models* directory along with base model information.

All training logs are logged in the saved directory under *saved/MODEL_NAME/RUN_ID/*.

The contents that get saved are:
- checkpoints at specific epochs
- output logs
- configuration data of the run
- model script used of the run

The codebase from *main.py* is presented to be operated via a **Command Line Interface (CLI)** provided by a framework built on top of *Pytorch Lightning*.

All training is logged to the **Weights and Biases** server: https://wandb.ai/monash-deep-neuron/microfluidics_summer?workspace

## How to use this framework?
To create the conda environment containing all the packages required, run the following command in the terminal, which assumes your device has an appropriate Conda package manager:

    conda env create --file environment.yml

To move into this environment, run the following command in the terminal:

    conda activate --n mfd
    
Acceptable commands in using the CLI can be found by running:

    python main.py --help

To train a model, you will need three things:
1. *model.py* 
   - A model class that represents the neural network being trained
   - The model **must** inherit from *models.base.BaseModel*
   - Only the ***\_\_init__()*** and the ***forward()*** method need to be implemented
2. *dataModule.py*
    - A data module class from Pytorch Lightning that houses all data hyperparameters, transforms, dataloaders, and path to the dataset.
3. *config.yml* - A configuration file of all the hyperparameters and directories specified for training, as well as the class path to the model and the data module themselves.

To train a model, run:

    python main.py fit --config config.yml

This will begin training with the hyperparameters and settings given on the supplied model using the supplied data module. Metrics are tracked in the WandB server.

## What debugging options are there for training?
There are **three** debugging configuration files that can be run in an order, to ensure that your given model and data module have indeed been implemented correctly and can train. **No results get logged** and **no model progress is checkpointed** during debugging.

Each debugging file has a purpose, and to use them, simply replace the *config.yml* with the corresponding debug file in the previous training (fit) command.

1. *debug_fastDevRun.yml* - Runs the model on the dataset for **one** training batch and **one** validation batch

        python main.py fit --config debug_fastDevRun.yml

2. *debug_overfit.yml* - Runs the model on only 5% of the dataset to see whether the model has the potential to even overfit the data, if it does not, it is not indicative of a good model

        python main.py fit --config debug_overfit.yml

3. *debug_profiler.yml* - Runs the model on only 50% of the dataset over 5 epochs to obtain statistics on whether there are any bottlenecks in the process by any function

        python main.py fit --config debug_profiler.yml

These files can also be configured differently to your specifications, such as using 1% of the data for overfitting instead.

## What metrics are logged?

Metrics are saved for training and validation separately. For each mode the following metrics are calculated and logged:
1. Accuracy
2. Recall
3. Precision
4. F1 Score

## Examples
### Example of Model Script

    from torch import nn
    import torch.nn.functional as F
    from typing import Callable
    from models import BaseModel

    class ExampleModel(BaseModel):
        """
        Description of Model
        """
        def __init__(self, learning_rate: float = 0.001, num_classes: int = 10, loss_fn: Callable = F.cross_entropy) -> None:
            super(ExampleModel, self).__init__(learning_rate, num_classes, loss_fn)

            # Required layers for model
            self.conv1 = nn.Sequential(  
                        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2) # (16,14,14)
                        )
            self.conv2 = nn.Sequential( # (16,14,14)
                        nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)
                        nn.ReLU(),
                        nn.MaxPool2d(2) # (32,7,7)
                        )
            self.out = nn.Linear(32*7*7, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1) # (batch, 32,7,7) -> (batch, 32*7*7)
            output = self.out(x)
            return output

### Example of Data Module Script
This example uses an MNIST dataset. For custom datasets, the *prepare_data()* hook might be required as well as a custom Pytorch Dataset class which implements the *\_\_init__()*, *\_\_len__()* and *\_\_getitem__()* calls.

    import pytorch_lightning as pl
    import torchvision.transforms as t
    from torchvision.datasets import MNIST
    from torch.utils.data import random_split, DataLoader

    class MNISTDataModule(pl.LightningDataModule):
        """
        Encapsulates the dataset to allow the Trainer to retrieve the appropriate dataloaders
        and apply relevant transforms.
        """
        def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
            """
            Sets up desired hyperparameters and directory to put data in.
            - data_dir: str - Directory where dataset is downloaded in and where to find data
            - batch_size: int - The desired loading batch size for dataloaders
            - num_workers: int - The number of workers given to each dataloader
            """
            super().__init__()
            # Save hyperparameters
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.num_workers = num_workers

        def setup(self, stage) -> None:
            # Create transforms
            transforms = t.Compose([t.ToTensor(), t.Normalize((0.1307,), (0.3081,))])

            # Download and instantiate dataset
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=transforms)

            # Split into training and validation
            self.mnist_train, self.mnist_val = random_split(mnist_full, [50000, 10000])

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

### Example of Configuration File Settings
Below is the configuration settings for training a **ExampleModel** with **MNISTDataModule**, saving the logs in *saved/*, with a **learning rate** of 0.001, loss function of **cross entropy**, **batch size** of 50, **saving top 5** model checkpoints as well as the last epoch model state, and only stopping training after **20 epochs of no improvement** on the **minimum validation/loss**, logging all metrics to Monash Deep Neruon's WandB server.

    # pytorch_lightning==1.8.3.post1
    seed_everything: 22
    trainer:
    logger:
        class_path: pytorch_lightning.loggers.WandbLogger
        init_args:
        project: microfluidics_summer
        entity: monash-deep-neuron
        offline: false
    enable_checkpointing: true
    callbacks:
        - class_path: pytorch_lightning.callbacks.EarlyStopping
        init_args:
            monitor: validation/loss
            mode: min
            patience: 20
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
        init_args:
            dirpath: saved
            filename: "epoch={epoch}-val_loss={validation/loss:.2f}-val_acc={validation/accuracy:.2f}"
            auto_insert_metric_name: false
            monitor: validation/accuracy
            mode: max
            save_top_k: 5
            save_last: true
        - class_path: pytorch_lightning.callbacks.DeviceStatsMonitor
        init_args:
            cpu_stats: false
    gpus: null
    check_val_every_n_epoch: 1
    max_epochs: -1
    enable_model_summary: true
    deterministic: true
    num_sanity_val_steps: 0
    model:
    class_path: models.ExampleModel.ExampleModel
    init_args:
        learning_rate: 0.001
        loss_fn: torch.nn.functional.cross_entropy
        num_classes: 10
    data:
    class_path: dataModules.MNISTDataModule.MNISTDataModule
    init_args:
        data_dir: data/
        batch_size: 50
        num_workers: 0
    ckpt_path: null

### How to resume from a checkpoint?
To resume from a checkpoint, simply edit the config.yml file to have:

    ckpt_path: path/to/checkpoint/file.ckpt
