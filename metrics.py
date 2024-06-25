import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Assuming you have a custom dataset class similar to `CustomDataset` provided
from datasets import CustomDataset  # Update with your actual dataset module

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to calculate precision and recall
def calculate_precision_recall(predictions, targets, iou_threshold=0.5):
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred_boxes, target_boxes in tqdm(zip(predictions, targets), total=len(predictions)):
        detected = []
        for pred_box in pred_boxes['boxes']:
            is_tp = False
            for target_box in target_boxes['boxes']:
                iou = calculate_iou(pred_box, target_box)
                if iou >= iou_threshold and tuple(target_box) not in detected:
                    true_positives[pred_boxes['labels'][pred_boxes['boxes'].tolist().index(pred_box)]] += 1
                    detected.append(tuple(target_box))
                    is_tp = True
                    break
            if not is_tp:
                false_positives[pred_boxes['labels'][pred_boxes['boxes'].tolist().index(pred_box)]] += 1

        for target_box in target_boxes['boxes']:
            if tuple(target_box) not in detected:
                false_negatives[target_boxes['labels'][target_boxes['boxes'].tolist().index(target_box)]] += 1

    # Compute precision and recall per class
    precision = {}
    recall = {}
    for label in set(list(true_positives.keys()) + list(false_negatives.keys())):
        precision[label] = true_positives[label] / float(true_positives[label] + false_positives[label] + 1e-6)
        recall[label] = true_positives[label] / float(true_positives[label] + false_negatives[label] + 1e-6)

    return precision, recall

# Load the model
model = torch.load('/content/outputs/best_model.pth')  # Load your model here
model.eval()

# Assuming you have a DataLoader for your test dataset
test_dataset = CustomDataset('/content/Retinanet-1/test', RESIZE_TO, RESIZE_TO, CLASSES, transforms=None)  # Update with your dataset parameters
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []
targets = []

# Run inference on test dataset
with torch.no_grad():
    for image, target in test_loader:
        image = image.cuda()  # Assuming CUDA is available
        output = model(image)
        predictions.append(output)
        targets.append(target)

# Calculate precision and recall
precision, recall = calculate_precision_recall(predictions, targets)

# Print precision and recall for each class
for label, prec in precision.items():
    print(f'Class: {label}, Precision: {prec:.4f}, Recall: {recall[label]:.4f}')
