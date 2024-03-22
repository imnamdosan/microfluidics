from pytorch_lightning.callbacks import Callback

class KeyboardCallback(Callback):
    """
    Keyboard callback that ensures that if training is ended manually prior to specified
    max epochs, the final confusion matrix results will still be printed.
    """
    def on_exception(self, trainer, pl_module, *args, **kwargs):
        if not trainer.current_epoch:
            return
        print("\nFinal Confusion Matrix Results:")
        print("Training:")
        print(pl_module.train_confusion_matrix)
        print("Validation:")
        print(pl_module.val_confusion_matrix)