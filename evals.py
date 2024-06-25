import torch
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics import IoU
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

# Evaluation function
def compute_precision_recall(predictions, targets, num_classes, iou_threshold=0.5):
    iou = IoU(num_classes=num_classes)

    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        target_boxes = target['boxes']
        target_labels = target['labels']

        iou_value = iou(pred_boxes, target_boxes, pred_labels, target_labels)

        # Compute true positives and false negatives
        for cls in range(num_classes):
            true_positives = (iou_value >= iou_threshold).logical_and(pred_labels == cls + 1).sum().item()
            false_negatives = (iou_value < iou_threshold).logical_and(target_labels == cls + 1).sum().item()

            if (pred_labels == cls + 1).any():
                precision[cls] += true_positives / (true_positives + (pred_labels == cls + 1).sum().item() - true_positives)
            if (target_labels == cls + 1).any():
                recall[cls] += true_positives / (true_positives + false_negatives)

    precision /= len(predictions)
    recall /= len(targets)

    return precision, recall

# Function to compute mAP
def compute_mAP(predictions, targets, num_classes):
    iou = IoU(num_classes=num_classes)
    ap = torch.zeros(num_classes)

    for cls in range(num_classes):
        class_predictions = [pred for pred, target in zip(predictions, targets) if (pred['labels'] == cls + 1).any()]
        class_targets = [target for pred, target in zip(predictions, targets) if (target['labels'] == cls + 1).any()]

        if len(class_predictions) == 0 or len(class_targets) == 0:
            continue

        pred_boxes = torch.cat([pred['boxes'] for pred in class_predictions], dim=0)
        pred_scores = torch.cat([pred['scores'] for pred in class_predictions], dim=0)
        target_boxes = torch.cat([target['boxes'] for target in class_targets], dim=0)

        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        iou_value = iou(pred_boxes, target_boxes)

        true_positives = (iou_value >= 0.5).sum().item()
        false_positives = len(pred_boxes) - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(target_boxes)

        ap[cls] = precision * recall

    return ap

if __name__ == '__main__':
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset('/content/Retinanet-1/test')
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    predictions = []
    targets = []

    # Iterate through the test loader
    for images, batch_targets in tqdm(test_loader, total=len(test_loader)):
        images = [image.to(DEVICE) for image in images]
        batch_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch_targets]

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

    num_classes = NUM_CLASSES

    # Compute precision and recall
    precision, recall = compute_precision_recall(predictions, targets, num_classes)

    # Compute mAP
    ap = compute_mAP(predictions, targets, num_classes)

    for cls in range(num_classes):
        print(f"Class {cls}:")
        print(f"\tPrecision: {precision[cls]:.4f}")
        print(f"\tRecall: {recall[cls]:.4f}")
        print(f"\tmAP: {ap[cls]:.4f}")

    print(f"mAP_50: {ap.mean():.4f}")
