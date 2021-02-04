import os
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path='./save'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.pre_best_epoch = None
        self.save_path = save_path
        self.best_acc = None

    def __call__(self, val_loss, val_acc, current_epoch, model, save=True):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_acc = val_acc
            if save:
                self.save_checkpoint(val_loss, current_epoch, model)
            self.pre_best_epoch = current_epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, current_epoch, model)
            self.pre_best_epoch = current_epoch
            self.counter = 0
            
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            
        if self.early_stop:
            print("Early stop.")
            print(f"Best validation accuracy: {self.best_acc:.2f}%")
            self.counter = 0
            self.best_score = None
            self.val_loss_min = np.Inf
            self.pre_best_epoch = None

    def save_checkpoint(self, val_loss, current_epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        out = os.path.join(self.save_path, f"checkpoint_{current_epoch}.tar")

        # To save a DataParallel model generically, save the model.module.state_dict().
        # This way, you have the flexibility to load the model any way you want to any device you want.
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
        else:
            torch.save(model.state_dict(), out)
            
        if self.pre_best_epoch is not None:
            pre_out = os.path.join(self.save_path, f"checkpoint_{self.pre_best_epoch}.tar")
            if os.path.isfile(pre_out):
                os.remove(pre_out)
