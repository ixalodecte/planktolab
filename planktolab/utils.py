import torch
import numpy as np
import datetime


def get_features_probas_labels(model, dataloader):
    model.eval()
    features = []
    probas = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            feat = model.model.forward_features(x)
            features.append(feat.cpu().numpy())
            probas.append(torch.softmax(model(x), dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    features = np.concatenate(features, axis=0)
    probas = np.concatenate(probas, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, probas, labels

def generate_default_path_name():
    now = datetime.datetime.now()
    return "run_" + now.strftime("%Y-%m-%d__%H-%M-%S")
    