import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from typing import Callable

class BaseModel(pl.LightningModule):
    """
    A base model all models MUST inherit. Encapsulates all metric information and saves
    hyperparameters to the training system. Also details the training and validation steps
    and any console printing for use convenience.
    """
    def __init__(self, num_classes: int, loss_fn: Callable, print_metrics: bool) -> None:
        super(BaseModel, self).__init__()

        # Saves args in a dictionary, self.hparams
        self.save_hyperparameters()

        # Save attributes
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.print_metrics = print_metrics

        # Metric tracking objects
        metrics = MetricCollection({
            "accuracy" : MulticlassAccuracy(num_classes=self.num_classes, average="weighted"),
            "recall" : MulticlassRecall(num_classes=self.num_classes, average="macro"),
            "precision" : MulticlassPrecision(num_classes=self.num_classes, average="weighted"),
            "F1" : MulticlassF1Score(num_classes=self.num_classes, average="weighted"),
        })

        # Create confusion matrix trackers
        self.train_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.val_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        # Create the metric collection for training and validation
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="validation/")

    def training_step(self, batch, batch_idx):
        # Get (input, label)
        inputs, labels = batch

        # Run model on input
        outputs = self(inputs)

        # Get loss from loss function and log it
        loss = self.loss_fn(outputs, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        # Update metrics
        self.train_metrics.update(outputs, labels)
        self.train_cm.update(outputs, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        # Get (input, label)
        inputs, labels = batch

        # Run model on input
        outputs = self(inputs)

        # Log loss
        loss = self.loss_fn(outputs, labels)
        self.log("validation/loss", loss, on_epoch=True, on_step=False)

        # Update metrics
        self.val_metrics.update(outputs, labels)
        self.val_cm.update(outputs, labels)

        return loss

    def training_epoch_end(self, outputs):
        # Get metrics for training epoch and log them
        out = self.train_metrics.compute()
        self.train_confusion_matrix = self.train_cm.compute()
        self.log_dict(out, on_epoch=True)

        # Reset metrics
        self.train_metrics.reset()
        self.train_cm.reset()

        if self.print_metrics:
            # Save metrics for later print
            self.train_accuracy = out["train/accuracy"]
            self.train_recall = out["train/recall"]
            self.train_precision = out["train/precision"]
            self.train_F1 = out["train/F1"]

            # Print training metrics
            print(f"\n\nTraining Accuracy: {self.train_accuracy:5.2%}, Recall: {self.train_recall:5.2%}, Precision: {self.train_precision:5.2%}, F1: {self.train_F1:5.2%}")
            print(self.train_confusion_matrix)

            # Print validation metrics
            print(f"Validation Accuracy: {self.val_accuracy:5.2%}, Recall: {self.val_recall:5.2%}, Precision: {self.val_precision:5.2%}, F1: {self.val_F1:5.2%}")
            print(self.val_confusion_matrix, end="\n\n")

    def validation_epoch_end(self, outputs):
        # Get metrics for training epoch and log them
        out = self.val_metrics.compute()
        self.val_confusion_matrix = self.val_cm.compute()
        self.log_dict(out, on_epoch=True)

        # Reset metrics
        self.val_metrics.reset()
        self.val_cm.reset()

        if self.print_metrics:
            # Save metrics for later print
            self.val_accuracy = out["validation/accuracy"]
            self.val_recall = out["validation/recall"]
            self.val_precision = out["validation/precision"]
            self.val_F1 = out["validation/F1"]

    def on_train_end(self) -> None:
        # Prints confusion matrices results for training and validation after final epoch
        print("\nFinal Confusion Matrix Results:")
        print("Training:")
        print(self.train_confusion_matrix)
        print("Validation:")
        print(self.val_confusion_matrix)