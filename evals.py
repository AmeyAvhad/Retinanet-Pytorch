import torch
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection import precision_recall, APObjectDetection
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

# Evaluation function
def validate(valid_data_loader, model):
    model.eval()

    num_classes = NUM_CLASSES
    precision_scores = torch.zeros(num_classes)
    recall_scores = torch.zeros(num_classes)
    ap = APObjectDetection(iou_threshold=0.5, num_classes=num_classes)

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        preds = []
        target = []

        for j in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[j]['boxes'].detach().cpu()
            true_dict['labels'] = targets[j]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[j]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[j]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[j]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

        # Calculate precision and recall per class
        precision, recall = precision_recall(preds, target, iou_threshold=0.5, num_classes=num_classes)
        precision_scores += precision
        recall_scores += recall

        # Update AP calculation
        ap.update(preds, target)

    # Compute mAP
    ap_summary = ap.compute()

    # Average precision and recall across all batches
    precision_scores /= len(valid_data_loader)
    recall_scores /= len(valid_data_loader)

    return precision_scores, recall_scores, ap_summary

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset('/content/Retinanet-1/test')
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    precision, recall, ap_summary = validate(test_loader, model)

    for idx in range(NUM_CLASSES):
        print(f"Class {idx}:")
        print(f"\tPrecision: {precision[idx]:.4f}")
        print(f"\tRecall: {recall[idx]:.4f}")

    print(f"mAP_50: {ap_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {ap_summary['map']*100:.3f}")
