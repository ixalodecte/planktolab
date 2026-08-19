# Project README

## Overview

This project provides a command-line interface (CLI) built with **Typer** for training, evaluating, and running deep learning models (mainly for computer vision tasks such as classification). It also includes utilities for dataset preprocessing, cross-validation training, anomaly detection, inference, and carbon emission tracking. We also implemented a graphical user interface to simply the usage of this tool.

---

# Installation

This guide explains how to install and use this project as a CLI tool using **pipx**.

---

## 1. Install pipx

:contentReference[oaicite:0]{index=0} allows you to install Python CLI tools in isolated environments without affecting your system Python.

### Install pipx

#### macOS / Linux

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```
Then restart your terminal.

#### Windows

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
```
Then restart your terminal.

#### Verify pipx installation

```bash
pipx --version
```

### 2. Install the CLI tool

```bash
pipx install git+https://github.com/ixalodecte/planktonlab.git
```

# CLI Usage

This project provides a command-line interface (CLI) for training, automatic label error detection, and inference of computer vision models.
The folders containing the image should be organized hierarchicaly. It means that there is a dedicated folder for each classes. Example:
```
train_folder
   - Asterionella
   - Ceratium
   - Keratella cochlearis
   ...

val_folder
...
```

---

## General Help

To display all available commands:

```bash
planktolab --help
```

## Resize Images

```bash
planktolab resize-images \
  data/input \
  data/output \
  --size 128
```

## Train a Model



If you already have a train, test and val folder:

```bash
planktolab train \
  your_train_folder \
  convnext_tiny \
  --val-path your_validation_folder \
  --test-path your_test_folder \
  --output-path outputs \
  --max-epoch 10 \
  --batch-size 64 \
  --image-size 128 \
```

If all your image are inside one dataset and you want to split it dynamicaly, set val-ratio and test-ratio:

```bash
planktolab train \
  data/train \
  convnext_tiny \
  --output-path outputs \
  --max-epoch 10 \
  --batch-size 64 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --image-size 128 \
```

## K-Fold label error detection
To detect potential mislabeled annotation in the dataset. First, kfold validation. The following command will run a k fold training on the input folder.

```bash 
planktolab run-kfold \
  data_path \
  convnext_tiny \
  --kfold 5 \
  --output-path output_kfold_path \
  --max-epoch 20 \
  --batch-size 64 \
  --image-size 128
```

Then run the label annotation error detection script
```bash
planktolab detect-suspect \
  output_kfold_path \
  outputs_path \
  --kfold 5 \
  --max-epoch 20 \
  --batch-size 64 \
  --image-size 128
```

# Graphical user interface

The graphical user interface (GUI) provides the same functionalities as the command-line client, but can be more user friendly.

But, the GUI may be more prone to bugs than the command-line client because the code behind it is more complex. If you encounter any issues, feel free to report them in the "Issues" section so that we can fix them.

To launch the GUI, run the following command in a terminal:

```planktolab-gui```

Then, open a web browser and copy-paste the URL displayed in the terminal (for example, http://127.0.0.1:8050).