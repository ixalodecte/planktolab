import torch
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support

import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import Counter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import Accuracy, F1Score
import logging

import numpy as np


def focal_loss_ce(inputs, targets, alpha=1, gamma=2):
    logpt = F.log_softmax(inputs, dim=1)
    pt = torch.exp(logpt)
    logpt = logpt.gather(1, targets.unsqueeze(1))
    pt = pt.gather(1, targets.unsqueeze(1))
    loss = -alpha * (1 - pt) ** gamma * logpt
    return loss.mean()

def get_log_weighted_CE(labels, num_classes, alpha=1.0):

    num_all = Counter(labels)

    distrib = np.array([num_all[i] for i in range(num_classes)])
    eps = 1e-6
    distrib = np.maximum(distrib, eps)
    weights = np.log1p(np.sum(distrib) / distrib) ** alpha
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float)
    )

    return criterion

class LightningWrapper(pl.LightningModule):
    def __init__(
            self, 
            criterion, 
            num_classes, 
            encoder, 
            class_names, 
            lr=1e-5, 
            max_epoch=20):
        super().__init__()
        self.model = encoder

        self.num_classes = num_classes
        self.class_names = class_names

        self.test_preds = []
        self.test_labels = []
        self.max_epoch = max_epoch


        self.criterion = criterion

        self.lr = lr

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Saver metrics
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": []
        }
        self.epoch = 0
        self.global_logger = logging.getLogger(__name__)
        self.global_logger.info("start training")
        

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self(x)
        loss = self.criterion(preds, y)
        _ = self.train_acc(preds, y)
        _ = self.train_f1(preds, y)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self(x)
        loss = self.criterion(preds, y)
        _ = self.val_acc(preds, y)
        _ = self.val_f1(preds, y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        self.log("train/acc", train_acc, prog_bar=True, sync_dist=True)
        self.log("train/f1", train_f1, prog_bar=True, sync_dist=True)

        self.train_acc.reset()
        self.train_f1.reset()
        self.epoch += 1

        self.global_logger.info(f"Epoch {self.epoch} - Train - Accuracy: {train_acc}, F1-Score: {train_f1}")



    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()


        self.log("val/acc", val_acc, prog_bar=True, sync_dist=True)
        self.log("val/f1", val_f1, prog_bar=True, sync_dist=True)
        self.global_logger.info(f"Validation - Accuracy: {val_acc}, F1-Score: {val_f1}")

        self.val_acc.reset()
        self.val_f1.reset()


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds).numpy()
        all_labels = torch.cat(self.test_labels).numpy()

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average=None,
            labels=np.arange(self.num_classes),
            zero_division=0
        )

        self.test_precision = precisions
        self.test_recall = recalls
        self.test_f1 = f1s
        self.test_supports = supports
        
        accuracy = np.mean(all_labels == all_preds)

        self.log("test/precision_macro", precisions.mean())
        self.log("test/recall_macro", recalls.mean())
        self.log("test/f1_macro", f1s.mean())
        self.log("test/accuracy", accuracy)

        self.test_results = {
            "precision_per_class": precisions,
            "recall_per_class": recalls,
            "f1_per_class": f1s,
            "support_per_class": supports,
            "accuracy": accuracy
        }

        self.test_preds.clear()
        self.test_labels.clear()
        self.global_logger.info(f"Test - Accuracy: {accuracy}, Precision (macro): {precisions.mean()}, Recall (macro): {recalls.mean()}, F1-Score (macro): {f1s.mean()}")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        #total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epoch,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    
    def get_loss(self):
        return self.history["train_loss"], self.history["val_loss"]


    def save_prec_rec_f1(self, path):
        """
        Save precision, recall, f1-score and support for each class in a CSV file.
        Columns: Class, Precision, Recall, F1-Score, Support
        """
        with open(path, "w") as f:
            f.write("Class,Precision,Recall,F1-Score,Support\n")
            for i in range(self.num_classes):
                class_name = self.class_names[i] if self.class_names else str(i)
                f.write(f"{class_name},{self.test_precision[i]:.4f},{self.test_recall[i]:.4f},{self.test_f1[i]:.4f},{self.test_supports[i]}\n")
