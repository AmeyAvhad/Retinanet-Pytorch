import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config  # Assuming config.py contains CLASSES and other configurations

# Function to get predicted and true labels
def get_predicted_true_labels(model, data_loader):
    predicted_labels = []
    true_labels = []

    model.eval()

    for images, batch_targets in data_loader:
        images = [image.to(config.DEVICE) for image in images]
        batch_targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in batch_targets]

        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, batch_targets):
            pred_labels = output['labels'].cpu().numpy()
            true_labels.extend(target['labels'].cpu().numpy())

            # If using softmax for classification, get the predicted class as the one with highest score
            # pred_labels = np.argmax(output['scores'].cpu().numpy(), axis=1)

            predicted_labels.extend(pred_labels)

    return predicted_labels, true_labels

# Function to plot confusion matrix using Seaborn
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == '__main__':
    model = create_model(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE).eval()

    valid_dataset = create_valid_dataset(config.VALID_DIR)
    valid_loader = create_valid_loader(valid_dataset, num_workers=config.NUM_WORKERS)

    predicted_labels, true_labels = get_predicted_true_labels(model, valid_loader)

    plot_confusion_matrix(true_labels, predicted_labels, classes=config.CLASSES[1:])  # Exclude '__background__'

