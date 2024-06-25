import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from config import CLASSES, NUM_CLASSES, DEVICE, RESIZE_TO
from custom_utils import get_valid_transform
from model import create_model
from datasets import CustomDataset
import argparse

# Function to calculate Average Precision (AP) for each class
def calculate_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_dir', required=True, help='Path to the test directory')
args = parser.parse_args()

# Load the model
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('/content/outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Prepare the dataset and dataloader
test_dataset = CustomDataset(args.test_dir, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Store all true boxes and predicted boxes
all_true_boxes = [[] for _ in range(NUM_CLASSES)]
all_pred_boxes = [[] for _ in range(NUM_CLASSES)]
all_scores = [[] for _ in range(NUM_CLASSES)]

with torch.no_grad():
    for images, targets in tqdm(test_loader):
        images = list(image.to(DEVICE) for image in images)
        outputs = model(images)
        
        for target, output in zip(targets, outputs):
            try:
                if isinstance(target, str):
                    # Handle case where target is unexpectedly a string
                    print(f"Skipping target: {target}")
                    continue
                
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                for i in range(len(true_boxes)):
                    box = true_boxes[i]
                    label = true_labels[i]
                    all_true_boxes[label].append(box)
                
                for i in range(len(pred_boxes)):
                    box = pred_boxes[i]
                    label = pred_labels[i]
                    score = scores[i]
                    all_pred_boxes[label].append(box)
                    all_scores[label].append(score)
            except Exception as e:
                print(f"Error processing target: {e}")
                print(f"Target type: {type(target)}")
                print(f"Target contents: {target}")
                continue

# Calculate AP for each class
aps = []
for i in range(1, NUM_CLASSES):  # Assuming class 0 is background
    true_boxes = np.array(all_true_boxes[i])
    pred_boxes = np.array(all_pred_boxes[i])
    scores = np.array(all_scores[i])
    
    if len(true_boxes) == 0 or len(pred_boxes) == 0:
        aps.append(0)
        continue
    
    sorted_indices = np.argsort(-scores)
    pred_boxes = pred_boxes[sorted_indices]
    scores = scores[sorted_indices]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    total_true_boxes = len(true_boxes)
    
    detected_boxes = []
    
    for j in range(len(pred_boxes)):
        pred_box = pred_boxes[j]
        ious = box_iou(torch.tensor(pred_box).unsqueeze(0), torch.tensor(true_boxes))
        max_iou, max_iou_index = ious.max(1)
        
        if max_iou > 0.5 and max_iou_index not in detected_boxes:
            tp[j] = 1
            detected_boxes.append(max_iou_index)
        else:
            fp[j] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / total_true_boxes
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ap = calculate_ap(recall, precision)
    aps.append(ap)

# Print the AP for each class and the mAP
for i in range(1, NUM_CLASSES):
    print(f"AP for {CLASSES[i]}: {aps[i-1]:.4f}")  # Note the index adjustment since aps does not include background class
mAP = np.mean(aps)
print(f"mAP: {mAP:.4f}")
