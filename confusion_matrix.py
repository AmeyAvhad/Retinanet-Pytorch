import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model
from config import (
    DEVICE, 
    NUM_CLASSES, 
    VALID_DIR,
    RESIZE_TO
)
from datasets import create_valid_loader, create_valid_dataset  # Import create_valid_dataset

from torchmetrics import ConfusionMatrix

def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot to the outputs folder
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    plt.show()

def evaluate(valid_data_loader, model):
    print('Evaluating')
    model.eval()
    
    target = []
    preds = []
    all_true_labels = []
    all_pred_labels = []
    
    for data in valid_data_loader:
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
            
            # Collect labels for confusion matrix
            all_true_labels.extend(true_dict['labels'])
            all_pred_labels.extend(preds_dict['labels'])
        #####################################

    # Compute confusion matrix manually
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int32)
    for true, pred in zip(all_true_labels, all_pred_labels):
        cm[true, pred] += 1

    cm = cm.numpy()
    
    return cm

if __name__ == '__main__':
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description='Evaluate the model on the validation set.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model.')
    args = parser.parse_args()

    # Load the trained model.
    model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # Create the validation data loader.
    valid_dataset = create_valid_dataset(VALID_DIR)  # Create valid dataset
    valid_loader = create_valid_loader(valid_dataset, num_workers=4)

    # Evaluate the model and get the confusion matrix.
    cm = evaluate(valid_loader, model)

    # Plot confusion matrix.
    class_names = [f'Class {i}' for i in range(NUM_CLASSES)]
    plot_confusion_matrix(cm, class_names, 'outputs')
