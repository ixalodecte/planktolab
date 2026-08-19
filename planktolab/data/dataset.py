import os

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
#from torchvision.io import read_image
#from torchvision.io import decode_image
import sklearn
import cv2


def get_val_augment(colored, size, normalize=True):
    if colored:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5]
        std = [0.2]

    base_transform = A.Compose([
        #A.ToGray(p=1),
        A.LongestMaxSize(size),
        A.PadIfNeeded(min_height=size, min_width=size,
                border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        A.Normalize(mean=mean, std=std, p=1.0 if normalize else 0),
        ToTensorV2()

    ])
    return base_transform


def resize(size):

    base_transform = A.Compose([
        A.LongestMaxSize(size),
    ])
    return base_transform



def get_train_augment(colored, size):
    if colored:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5]
        std = [0.2]
    augmented_transform = A.Compose([
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5 if colored else 0),
        A.GaussianBlur(blur_limit=(5, 15), sigma_limit=(0.5, 5), p=0.5 ),
        A.LongestMaxSize(size),
        #A.RandomResizedCrop(
        #    size=(224, 224),
        #    scale=(0.8, 1.0), ratio=(1,1), p=0.5
        #),
        #A.RandomResizedCrop((224, 224), scale=(0.7, 1.0), ratio=(0.2, 5)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1),
        #A.ToGray(p=1),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.PadIfNeeded(min_height=size, min_width=size,
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        A.CoarseDropout(p=0.5, num_holes_range=(5,15), hole_height_range=(0.1,0.2), hole_width_range=(0.1,0.2)),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()

    ])
    return augmented_transform


def normalize_added_features(features):
    if features is None:
        return None
    return (features - np.mean(features)) / np.std(features)

class ImageDataset(Dataset):
    def __init__(self, X, labels, transform=None, colored=False):

        self.images = X
        self.labels = labels
        self.transform = transform

        self.colored = colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.colored:
            image = cv2.imread(self.images[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Image not found or corrupted: {self.images[idx]}")
        
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = -1

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        #image = torch.from_numpy(image)
        if not self.colored:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        #else:
        #    image = image.permute(2, 0, 1)
        


        return image, torch.tensor(label, dtype=torch.long)

def images_labels_from_hierarchie(path):
    images_fnames = []
    labels = []

    classes = sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        folder = os.path.join(path, cls)
        files = os.listdir(folder)

        images_fnames.extend([os.path.join(folder, f) for f in files])
        labels.extend([class_to_idx[cls]] * len(files))
    

    images_fnames = np.array(images_fnames, dtype=object)
    labels = np.array(labels)

    asort = np.argsort(images_fnames)
    images_fnames = images_fnames[asort]
    labels = labels[asort]


    return images_fnames, labels, classes

def images_from_folder(path):
    images_fnames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                images_fnames.append(os.path.join(root, file))
    return np.array(images_fnames, dtype=object)

def kfold_stratified(images_fnames, labels, n_fold):
    indices = np.arange(len(labels))
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    folds = []
    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx, val_idx))

    return folds
