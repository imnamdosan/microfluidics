import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, F1Score, Recall, Precision, MetricCollection
from typing import Callable

class BaseModel_old(pl.LightningModule):
    """
    A base model all models MUST inherit. Encapsulates all metric information and saves
    hyperparameters to the training system. Also details the training and validation steps
    and any console printing for use convenience.
    """
    def __init__(self, learning_rate: float, num_classes: int, loss_fn: Callable) -> None:
        super(BaseModel_old, self).__init__()

        # Saves args in a dictionary, self.hparams
        self.save_hyperparameters()

        # Save attributes
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_fn = loss_fn

        # Metric tracking objects
        metrics = MetricCollection({
            "accuracy" : Accuracy(task="multiclass", num_classes=self.num_classes),
            "F1" : F1Score(task="multiclass", num_classes=self.num_classes),
            #"recall" : Recall(task=self.task, num_classes=self.num_classes),
            #"precision" : Precision(task=self.task, num_classes=self.num_classes)
        })
        
        # Create the metric collection for training and validation
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="validation/")

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # Get (input, label)
        input, label = batch

        # Run model on input
        output = self(input)

        # Get loss from loss function and log it
        loss = self.loss_fn(output, label)
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        # Update and log metrics
        output = self.train_metrics(output, label)
        self.log_dict(output, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # Get (input, label)
        input, label = batch

        # Run model on input
        output = self(input)

        # Log loss
        loss = self.loss_fn(output, label)
        self.log("validation/loss", loss, on_epoch=True, on_step=False)

        # Update and log metrics
        output = self.val_metrics(output, label)
        self.log_dict(output, on_epoch=True, on_step=False)

        return loss

    # def on_validation_epoch_end(self) -> None:
    #     super().on_validation_epoch_end()

        # Compute metrics and print training accuracy
        # train_results = self.train_metrics.compute()
        # print("\n\nTraining Accuracy: {:5.2f}%".format(train_results["train/accuracy"]*100))

        # # Compute metrics and print validation accuracy
        # val_results = self.val_metrics.compute()
        # print("Validation Accuracy: {:5.2f}%\n".format(val_results["validation/accuracy"]*100))