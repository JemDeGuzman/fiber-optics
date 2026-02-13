import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Import the model class from your training script
# Ensure FiberClassifier and HierarchicalEncoder are defined or imported here
from fiber_classification import FiberClassifier, FiberDataset 

def evaluate_model(model_path, csv_file, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.CenterCrop(300), # 👈 Focus on the fiber, ignore the edges
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = FiberDataset(csv_file, data_dir, transform=transform)
    # Using a small batch size for evaluation to ensure stability
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 2. Load Model
    model = FiberClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        for l, r, t, labels in loader:
            l, r, t, labels = l.to(device), r.to(device), t.to(device), labels.to(device)
            
            outputs = model(l, r, t)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Metrics Calculation
    target_names = ['Abaca', 'Daratex']
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 4. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Fiber Classification Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_model(
        model_path='fiber_model.pth', 
        csv_file='db_export.csv', 
        data_dir='./labeled_dataset'
    )