from pytorch_lightning.cli import LightningCLI
from datetime import datetime
from os import makedirs
from shutil import copy
import importlib
import wandb

class CustomLightningCLI(LightningCLI):
    """
    Abstraction layer on LightningCLI to edit directory paths and to customise where
    logging information gets sent, and what extra information is saved per run.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse_arguments(self, parser, args) -> None:
        """
        Parses the supplied .yaml file and edits self.config for desired paths based on parsed
        results in other sections such as model name. Ensures a new directory per run based on timestamp
        is created to put all run information and checkpoints in.
        """
        super().parse_arguments(parser, args)
        # If tuning, then do nothing
        if self.config['subcommand'] == "tune":
            self.save_config_callback = None
            return

        # If in debug mode, no checkpointing or files need to be saved
        if not self.config['fit']['trainer']['enable_checkpointing']:
            self.save_config_callback = None
            return

        # Find saving directory and name of model being trained
        dirpath = self.config['fit']['trainer']['callbacks'][1]['init_args']['dirpath']
        lstOfStr = self.config['fit']['model']['class_path'].split('.') # Gets last string in "models.ModelName.ModelName"
        modelName = lstOfStr[-1]

        # Create Run ID with timestamp
        d = datetime.now()
        dateString = f"{d.year:4d}-{d.month:02d}-{d.day:02d}__{d.hour:02d}-{d.minute:02d}-{d.second:02d}"
        runID =  'Run_ID__' + dateString

        # Create directory for checkpoints to be saved and for configuration data
        ckptDir = f"{dirpath}/{modelName}/{runID}/checkpoints/"
        configDir = f"{dirpath}/{modelName}/{runID}/"
        makedirs(ckptDir)   # Create the new directory and any superfolders required
        wandbLabel = f"{modelName}__{dateString}"   # Create label for the run in WandB server

        # Overwrite with desired paths and labels
        self.config['fit']['trainer']['callbacks'][1]['init_args']['dirpath'] = ckptDir
        self.config['fit']['trainer']['logger']['init_args']['save_dir'] = configDir
        self.config['fit']['trainer']['logger']['init_args']['name'] = wandbLabel
        self.config['fit']['trainer']['logger']['init_args']['version'] = wandbLabel

        # Make copy of model.py into the Run_ID directory
        modelFileName = lstOfStr[-2] + ".py"
        modelFilePath = f"models/{modelFileName}"
        copy(modelFilePath, configDir)

        # Loging to WandB
        wandb.login(anonymous="never", key="75d3d7ba6635f3698b07f282461ff4e09a78695a", force=True)

    def instantiate_classes(self) -> None:
        """
        Custom class to parse a list of transforms into a Compose object in the datamodule.
        """
        super().instantiate_classes()
        dm = self.datamodule

        # Parse training transforms
        dictOfTransforms = dm.train_transforms.transforms
        train_transforms = [0]*len(dictOfTransforms)    # Create space
        for i in range(len(dictOfTransforms)):
            # Get transform
            dict = dictOfTransforms[i]

            # Get class of transform and import it
            class_path = dict["class_path"].split('.')
            class_name = class_path[-1]
            module_path = '.'.join(class_path[:len(class_path) - 1])
            module = importlib.import_module(module_path)

            # Instantiate class and save
            class_fn = getattr(module, class_name)
            kwargs = dict["init_args"]
            if kwargs is not None:
                train_transforms[i] = class_fn(**kwargs)
            else:
                train_transforms[i] = class_fn()
        
        # Parse validation transforms
        dictOfTransforms = dm.val_transforms.transforms
        val_transforms = [0]*len(dictOfTransforms)
        for i in range(len(dictOfTransforms)):
            # Get transform
            dict = dictOfTransforms[i]

            # Get class of transform and import it
            class_path = dict["class_path"].split('.')
            class_name = class_path[-1]
            module_path = '.'.join(class_path[:len(class_path) - 1])
            module = importlib.import_module(module_path)

            # Instantiate class and save
            class_fn = getattr(module, class_name)
            kwargs = dict["init_args"]
            if kwargs is not None:
                val_transforms[i] = class_fn(**kwargs)
            else:
                val_transforms[i] = class_fn()

        # Update transforms
        dm.train_transforms.transforms = train_transforms
        dm.val_transforms.transforms = val_transforms
        return