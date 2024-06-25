import torch
from tqdm import tqdm
import config  # Importing config.py file to access CLASSES
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

# Function to compute Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    intersection_top_left = torch.max(box1[:, None, :2], box2[:, :2])
    intersection_bottom_right = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    # Calculate intersection area
    intersection_area = torch.prod(intersection_bottom_right - intersection_top_left, dim=2) * (intersection_top_left < intersection_bottom_right).all(dim=2)
    
    # Calculate box areas
    box1_area = torch.prod(box1[:, 2:] - box1[:, :2], dim=1)
    box2_area = torch.prod(box2[:, 2:] - box2[:, :2], dim=1)
    
    # Calculate union area
    union_area = box1_area[:, None] + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# Function to compute precision, recall, and mAP per class
def compute_metrics(predictions, targets, num_classes, class_names, iou_threshold=0.5):
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    ap = torch.zeros(num_classes)

    for cls in range(num_classes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_targets = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'][pred['labels'] == cls]
            pred_scores = pred['scores'][pred['labels'] == cls]
            target_boxes = target['boxes'][target['labels'] == cls]

            total_targets += len(target_boxes)

            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                false_negatives += len(target_boxes)
                continue

            iou = calculate_iou(pred_boxes, target_boxes)
            if iou.numel() > 0:  # Check if iou tensor is not empty
                max_iou, _ = iou.max(dim=1)
                true_positives += (max_iou >= iou_threshold).sum().item()
                false_positives += (max_iou < iou_threshold).sum().item()
                false_negatives += (max_iou < iou_threshold).sum().item()

        # Compute precision and recall
        if true_positives + false_positives > 0:
            precision[cls] = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives > 0:
            recall[cls] = true_positives / (true_positives + false_negatives)

        # Compute AP (Average Precision)
        if true_positives > 0:
            ap[cls] = precision[cls] * recall[cls]

    return precision, recall, ap

if __name__ == '__main__':
    model = create_model(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE).eval()

    test_dataset = create_valid_dataset(config.VALID_DIR)
    test_loader = create_valid_loader(test_dataset, num_workers=config.NUM_WORKERS)

    predictions = []
    targets = []

    # Iterate through the test loader
    for images, batch_targets in tqdm(test_loader, total=len(test_loader)):
        images = [image.to(config.DEVICE) for image in images]
        batch_targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in batch_targets]

        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, batch_targets):
            predictions.append({
                'boxes': output['boxes'].cpu(),
                'scores': output['scores'].cpu(),
                'labels': output['labels'].cpu()
            })
            targets.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu()
            })

    num_classes = config.NUM_CLASSES
    class_names = config.CLASSES  # Use CLASSES from config.py

    # Compute precision, recall, and mAP
    precision, recall, ap = compute_metrics(predictions, targets, num_classes, class_names)

    for cls in range(num_classes):
        print(f"{class_names[cls]}:")
        print(f"\tPrecision: {precision[cls]:.4f}")
        print(f"\tRecall: {recall[cls]:.4f}")
        print(f"\tmAP: {ap[cls]:.4f}")

    print(f"mAP_50: {ap.mean():.4f}")
