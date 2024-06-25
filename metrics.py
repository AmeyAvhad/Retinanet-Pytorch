import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import matplotlib.pyplot as plt

from model import create_model
from datasets import CustomDataset
from config import (
    NUM_CLASSES, DEVICE, CLASSES, VALID_DIR
)

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from sklearn.metrics import precision_score, recall_score

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--threshold',
    default=0.5,
    type=float,
    help='detection threshold'
)
args = vars(parser.parse_args())

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Define the validation dataset and data loader.
valid_dataset = CustomDataset(
    VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=False
)

# Lists to store ground truth and prediction values.
true_boxes = []
true_labels = []
pred_boxes = []
pred_scores = []
pred_labels = []

# Iterate through the validation dataset.
for images, targets in valid_loader:
    # Move images and targets to device.
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    
    # Forward pass through the model.
    with torch.no_grad():
        outputs = model(images)

    # Post-process outputs to get predictions.
    for i in range(len(images)):
        true_boxes.append(targets[i]['boxes'].cpu().numpy())
        true_labels.append(targets[i]['labels'].cpu().numpy())

        boxes = outputs[i]['boxes'].detach().cpu().numpy()
        scores = outputs[i]['scores'].detach().cpu().numpy()
        labels = outputs[i]['labels'].detach().cpu().numpy()

        # Apply detection threshold.
        boxes = boxes[scores >= args['threshold']]
        labels = labels[scores >= args['threshold']]
        scores = scores[scores >= args['threshold']]

        pred_boxes.append(boxes)
        pred_scores.append(scores)
        pred_labels.append(labels)

# Flatten lists for computing precision and recall.
true_boxes_flat = np.concatenate(true_boxes, axis=0)
true_labels_flat = np.concatenate(true_labels, axis=0)
pred_boxes_flat = np.concatenate(pred_boxes, axis=0)
pred_scores_flat = np.concatenate(pred_scores, axis=0)
pred_labels_flat = np.concatenate(pred_labels, axis=0)

# Compute precision and recall for each class.
precision = precision_score(true_labels_flat, pred_labels_flat, average=None)
recall = recall_score(true_labels_flat, pred_labels_flat, average=None)

# Print precision and recall for each class.
for idx, cls in enumerate(CLASSES):
    if cls == '__background__':
        continue
    print(f'Class: {cls}')
    print(f'Precision: {precision[idx]:.4f}')
    print(f'Recall: {recall[idx]:.4f}')
    print('-' * 20)

# Optionally, you can compute mAP using tools like sklearn or specific metrics libraries.

# Example code for computing mAP with sklearn
# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(true_labels_flat, pred_scores_flat)

# Print mAP if needed.
# print(f'mAP: {average_precision:.4f}')
