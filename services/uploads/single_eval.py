import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure these match the definitions in your new training script
from single_test import SinglePropertyClassifier, FiberDataset 

def evaluate_individual_models(csv_file, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = FiberDataset(csv_file, data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Define the three models to test
    # 'index' refers to the position of l, r, t in the dataset return: (0, 1, 2)
    models_to_test = [
        {'name': 'Luster', 'file': 'luster_model.pth', 'index': 0},
        {'name': 'Roughness', 'file': 'roughness_model.pth', 'index': 1},
        {'name': 'Tensile', 'file': 'tensile_model.pth', 'index': 2}
    ]

    target_names = ['Abaca', 'Daratex']

    for m_info in models_to_test:
        if not os.path.exists(m_info['file']):
            print(f"[SKIP] {m_info['file']} not found.")
            continue

        print(f"\n--- Evaluating Individual Model: {m_info['name']} ---")
        
        # Load the specific model
        model = SinglePropertyClassifier().to(device)
        model.load_state_dict(torch.load(m_info['file'], map_location=device))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in loader:
                # data[0]=l, data[1]=r, data[2]=t, data[3]=labels
                input_img = data[m_info['index']].to(device)
                labels = data[3].to(device)
                
                outputs = model(input_img)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 1. Print Text Report
        print(classification_report(all_labels, all_preds, target_names=target_names))

        # 2. Confusion Matrix Visualization
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{m_info['name']} Model Performance')
        plt.show()

if __name__ == "__main__":
    evaluate_individual_models(
        csv_file='db_export.csv', 
        data_dir='./labeled_dataset'
    )