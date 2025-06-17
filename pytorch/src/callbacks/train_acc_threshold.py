# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from lightning import Callback


class TrainAccuracyThresholdCallback(Callback):
    """
    PyTorch Lightning callback that stops training when the current task's
    training accuracy exceeds a specified threshold.
    """
    
    def __init__(self, accuracy_threshold=0.9, metric_name="train/current_task_class_acc"):
        """
        Initialize the callback.
        
        Args:
            accuracy_threshold (float): The accuracy threshold at which to stop training.
                                       Default is 0.9 (90%).
            metric_name (str): The name of the accuracy metric to monitor.
                              Default is "train/current_task_class_acc".
        """
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.metric_name = metric_name
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Check if the train accuracy exceeds the threshold at the end of each training epoch.
        If it does, stop training.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The Lightning module being trained.
        """
        # Check if the metric exists in the logged metrics
        if self.metric_name in trainer.callback_metrics:
            current_acc = trainer.callback_metrics[self.metric_name].item()
            
            # If accuracy exceeds threshold, stop training
            if current_acc >= self.accuracy_threshold:
                print(f"\nStopping training as {self.metric_name} reached {current_acc:.4f}, "
                      f"which exceeds threshold of {self.accuracy_threshold:.4f}")
                trainer.should_stop = True
