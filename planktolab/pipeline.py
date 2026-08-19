from planktolab.data.dataset import get_train_augment, get_val_augment, kfold_stratified, images_labels_from_hierarchie, ImageDataset, images_from_folder, resize
from planktolab.models.backbone import create_model, get_available_models, save_model, load_model, ClassificationModel
from planktolab.utils import get_features_probas_labels, generate_default_path_name
from planktolab.models.trainer import LightningWrapper, get_log_weighted_CE
from planktolab.suspect_samples import self_confidence, normalized_margin, score_knn
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from torch.utils.data import DataLoader

import torch

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
import os

import pytorch_lightning as pl

import logging
import sys

import cv2
from enum import Enum
import shutil
from pathlib import Path




@dataclass
class DetectionConfig:
    input_path: str
    model_name: str
    kfold: int = 5
    output_path: str = "planktonlab_output/"
    max_epoch: int = 20



def run_resize_images(input_path, output_path, size):
    images, labels, classes = images_labels_from_hierarchie(input_path)

    os.makedirs(output_path, exist_ok=True)

    transform = resize(size)
    dataset = ImageDataset(images, labels, transform=transform, colored=True)

    for class_name in classes:
        os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

    idx = 0
    with torch.no_grad():
        for img, _ in dataset:
            label = labels[idx]
            class_name = classes[label]

            path = os.path.join(output_path, class_name, f"image_{idx}.jpg")

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(path, img)
            idx += 1
        

def run_train_kfolds(input_path, model_name, kfold, output_path, max_epoch, batch_size, size=128):
    # Load data
    if output_path is None:
        output_path = generate_default_path_name()
    os.makedirs(output_path)
    for i in range(kfold):
        os.makedirs(os.path.join(output_path, f"fold_{i + 1}"))

    images, labels, classes = images_labels_from_hierarchie(input_path)

    train_augment = get_train_augment(colored=True, size=size)
    test_augment = get_val_augment(colored=True, size=size)
    
    # Create model
    num_classes = len(classes)

    folds = kfold_stratified(images, labels, kfold)

    np.save(f"{output_path}/fname.npy", images)
    np.save(f"{output_path}/classes.npy", classes)

    # Train with kfold validation
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(len(images))
        print(max(train_idx), max(val_idx))
        print(min(train_idx), min(val_idx))
        print(f"Fold {fold + 1}/{kfold}")
        #model = create_model(model_name, num_classes)
        model = ClassificationModel(model_name, num_classes)

        train_set = ImageDataset([images[i] for i in train_idx], [labels[i] for i in train_idx], transform=train_augment, colored=True)
        val_set = ImageDataset([images[i] for i in val_idx], [labels[i] for i in val_idx], transform=test_augment, colored=True)

        # Create dataloaders

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


        # Train model
        crit = get_log_weighted_CE(train_set.labels, num_classes)

        model = LightningWrapper(
            criterion=crit,
            num_classes=num_classes,
            encoder=model, 
            class_names=classes, 
            lr=1e-5, 
            max_epoch=20
        )

        trainer = pl.Trainer(
            logger=False,
            max_epochs=max_epoch,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            #strategy="dp",
            #strategy="ddp_notebook",
            precision="16-mixed",  # mixed precision
            #callbacks=[early_stopping, checkpoint, lr_monitor],
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_checkpointing=False,
        )

        trainer.fit(model, train_loader)

        # Save model
        save_model(trainer.model, f"{output_path}/model_fold_{fold + 1}.pth")
        
        # test on val and save features and probas
        features, probas, labels2 = get_features_probas_labels(model, val_loader)

        np.save(f"{output_path}/fold_{fold + 1}/features.npy", features)
        np.save(f"{output_path}/fold_{fold + 1}/probas.npy", probas)
        np.save(f"{output_path}/fold_{fold + 1}/labels.npy", labels2)
        #np.save(f"{output_path}/fold_{fold + 1}/classes.npy", classes)
        np.save(f"{output_path}/fold_{fold + 1}/train_idx.npy", train_idx)
        np.save(f"{output_path}/fold_{fold + 1}/val_idx.npy", val_idx)
    

def run_detect_suspect(kfold_folder, output_path, method, threshold):
    fnames = np.load(f"{kfold_folder}/fname.npy", allow_pickle=True)
    classes = np.load(f"{kfold_folder}/classes.npy", allow_pickle=True)
    os.makedirs(output_path, exist_ok=True)
    image_suspect = []
    label_suspect = []
    score_suspect = []

    kfold = len([
        p for p in Path(kfold_folder).iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    ])
    
    for fold in range(kfold):
        print(f"fold {fold}")

        #images, labels, classes = images_labels_from_hierarchie(output_path)
        features = np.load(f"{kfold_folder}/fold_{fold + 1}/features.npy")
        probas = np.load(f"{kfold_folder}/fold_{fold + 1}/probas.npy")
        labels = np.load(f"{kfold_folder}/fold_{fold + 1}/labels.npy")
        val_idx = np.load(f"{kfold_folder}/fold_{fold + 1}/val_idx.npy")

        if method == "self_confidence":
            scores = self_confidence(labels, probas)
        elif method == "knn":
            scores = score_knn(features, labels, n_neighbors=11)
        elif method == "normalized_margin":
            scores = normalized_margin(labels, probas)
        else:
            raise NotImplementedError

        image_threshold = fnames[val_idx][scores < threshold]
        label_threshold = labels[scores < threshold]
        score_threshold = scores[scores < threshold]
        image_suspect.extend(image_threshold)
        label_suspect.extend(label_threshold)
        score_suspect.extend(score_threshold)
    

    score_suspect = np.array(score_suspect)
    asort = np.argsort(score_suspect)
    score_suspect = score_suspect[asort]
    image_suspect = np.array(image_suspect, dtype=object)[asort]
    label_suspect = np.array(label_suspect)[asort]

    
    with open(f"{output_path}/suspect_samples_{method}.txt", "w") as f:
        f.write("filename,label,score\n")
        for a,b,c in zip(image_suspect, label_suspect, score_suspect):
            f.write(f"{a},{classes[b]},{c}\n")

    #return image_suspect, label_suspect


def display_images_transform_n_times(images, labels, classes, transform, number=5):
    for i in range(number):
        augmented = transform(image=images[0])
        image = augmented["image"]

        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

        plt.imshow(image)
        plt.title(f"Label: {classes[labels[0]]}")
        plt.axis("off")
        plt.show()

def run_train(train_path, model_name, output_path, max_epoch, batch_size, val_path=None, test_path=None, val_ratio=0.2, test_ratio=0.2, image_size=128, alpha=1.0):

    print("batch_size", batch_size)

    if output_path is None:
        output_path = generate_default_path_name()
        print("generated")
        print(output_path)

    os.makedirs(output_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_path}/log.txt"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.info("TEST LOG")

    train_images, train_labels, classes = images_labels_from_hierarchie(train_path)

    if test_path is not None:
        test_images, test_labels, _ = images_labels_from_hierarchie(test_path)
    else:
        # train test split
        train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=test_ratio, stratify=train_labels, random_state=42)

    if val_path is not None:
        val_images, val_labels, _ = images_labels_from_hierarchie(val_path)
    else:
        # train val split
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_ratio, stratify=train_labels, random_state=42)
    

    train_augment = get_train_augment(colored=True, size=image_size)
    test_augment = get_val_augment(colored=True, size=image_size)
    
    # Create model
    num_classes = len(classes)
    model = create_model(model_name, num_classes)
    train_set = ImageDataset(train_images, train_labels, transform=train_augment, colored=True)
    val_set = ImageDataset(val_images, val_labels, transform=test_augment, colored=True)
    test_set = ImageDataset(test_images, test_labels, transform=test_augment, colored=True)



    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    crit = get_log_weighted_CE(train_set.labels, num_classes, alpha=alpha)

    early_stopping = EarlyStopping(
        monitor="val/acc",
        patience=10,
        mode="max"
    )


    checkpoint = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_acc:.4f}"
    )

    model = LightningWrapper(
        criterion=crit,
        num_classes=num_classes,
        encoder=model, 
        class_names=classes, 
        lr=1e-4, 
        max_epoch=max_epoch
    )

    trainer = pl.Trainer(
        logger=False,
        max_epochs=max_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        #strategy="dp",
        #strategy="ddp_notebook",
        precision="16-mixed",  # mixed precision
        #callbacks=[early_stopping, checkpoint, lr_monitor],
        callbacks=[early_stopping, checkpoint],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    test_results = model.test_results
    train_loss, val_loss = model.get_loss()

    test_results["train_loss"] = train_loss
    test_results["val_loss"] = val_loss

    save_model(model.model, f"{output_path}/{model_name}_model.pth")

    np.save(f"{output_path}/classes.npy", classes)
    np.save(f"{output_path}/train_loss.npy", train_loss)
    np.save(f"{output_path}/val_loss.npy", val_loss)

    file_to_write = "precision, recall, f1, support\n"
    for cls in classes:
        precision = test_results["precision_per_class"][classes.index(cls)]
        recall = test_results["recall_per_class"][classes.index(cls)]
        f1 = test_results["f1_per_class"][classes.index(cls)]
        support = test_results["support_per_class"][classes.index(cls)]
        file_to_write += f"{cls}, {precision}, {recall}, {f1}, {support}\n"
    
    with open(f"{output_path}/test_results_per_class.txt", "w") as f:
        f.write(file_to_write)


    return test_results


# enum output type
class InferenceOutputType(Enum):
    folder_simlink = "folder_simlink"
    folder_copy = "folder_copy"
    csv = "csv"
    #proba = "proba"

def run_inference(model_path, path, output_path=None, image_size=128, output_type="csv"):
    if output_type not in InferenceOutputType:
        raise ValueError(f"output_type must be one of {InferenceOutputType}")

    files = os.listdir(model_path)
    model_weights = [f for f in files if f.endswith(".pth")][0]
    model_name = model_weights.split("_model.pth")[0]
    if "classes.npy" in files:
        classes = np.load(os.path.join(model_path, "classes.npy"))
    else:
        raise RuntimeError("classes.npy not found in model path")
    

    model = create_model(model_name, num_classes=len(classes))  # num_classes should be the same as during training
    model = load_model(model, os.path.join(model_path, model_weights))
    model.eval()



    image = images_from_folder(path)
    transform = get_val_augment(colored=True, size=image_size)

    dataset = ImageDataset(image, labels=None, transform=transform, colored=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    pred_labels_all = []
    with torch.no_grad():
        for batch in dataloader:
            print("batch")
            image = batch[0]
            outputs = model(image)
            probas = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probas, dim=1)
            pred_label = pred_labels.cpu().numpy()
            pred_labels_all.append(pred_label)
    
        pred_labels_all = np.concatenate(pred_labels_all, axis=0)
    
    fnames = image

    
    if output_type == InferenceOutputType.csv:
        headers = "filename,label\n"
        rows = [f"{fname},{classes[pred_label]}" for fname, pred_label in zip(fnames, pred_labels_all)]
        with open(os.path.join(output_path, "inference_results.csv"), "w") as f:
            f.write(headers)
            f.write("\n".join(rows))

    elif output_type == InferenceOutputType.folder_simlink:
        for filename, pred_label in zip(fnames, pred_labels_all):
            class_name = classes[pred_label]
            class_folder = os.path.join(output_path, class_name)
            os.makedirs(class_folder, exist_ok=True)
            src = os.path.join(path, filename)
            dst = os.path.join(class_folder, filename)
            os.symlink(src, dst)

    elif output_type == InferenceOutputType.folder_copy:
        for filename, pred_label in zip(fnames, pred_labels_all):
            class_name = classes[pred_label]
            class_folder = os.path.join(output_path, class_name)
            os.makedirs(class_folder, exist_ok=True)
            src = os.path.join(path, filename)
            dst = os.path.join(class_folder, filename)
            shutil.copy(src, dst)

    return pred_labels_all, classes