import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import os
from PIL import Image
import glob
from math import pi
import time

from new_train import CAILN_Model, FiberDataset, PHEFM_Model, get_cnn_model

# --- 1. Fix the Class List Globally ---
# Ensure this matches your training exactly!
GLOBAL_CLASSES = ['Abaca', 'Daratex', 'Mixed']

# ... [Keep your Model Classes (CAILN, PHEFM, etc.) as they are] ...
def plot_radar_chart(df_all, classes):
    """Generates a spider plot comparing architectures on the 'All' feature set."""
    categories = ['Accuracy'] + classes
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # High-contrast palette for your paper
    colors = ['#E74C3C', '#2ECC71', '#3498DB'] 
    
    for i, (idx, row) in enumerate(df_all.iterrows()):
        values = [row['Accuracy'], row['Abaca_F1'], row['Daratex_F1'], row['Mixed_F1']]
        values += values[:1] # Close the circle
        
        ax.plot(angles, values, color=colors[i], linewidth=2, label=row['Architecture'], marker='o')
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0.8, 1.05) # Focus on high-performance range
    plt.title("Architectural Comparison: Merged Sensors (All)", size=16, y=1.1)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
    plt.savefig("radar_chart_final.png", bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_all_models(data_root, model_dir="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architectures = ['CNN', 'CAILN', 'PHEFM']
    features = ['luster_map', 'roughness_proxy', 'tensile_map', 'all']
    
    performance_data = []

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for arch_name in architectures:
        for feat_name in features:
            # Match your new naming convention
            model_path = os.path.join(model_dir, f"best_{arch_name}_{feat_name}_mixed.pth")
            if not os.path.exists(model_path):
                print(f"Skipping {model_path} - not found.")
                continue

            print(f"Testing: {arch_name} on {feat_name}...")
            
            val_set = FiberDataset(os.path.join(data_root, 'val'), feature_type=feat_name, transform=val_transform)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False) # Batch 1 for latency check

            if arch_name == 'CNN': model = get_cnn_model(num_classes=3)
            elif arch_name == 'CAILN': model = CAILN_Model(num_classes=3)
            elif arch_name == 'PHEFM': model = PHEFM_Model(num_classes=3)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            all_preds, all_labels = [], []
            latencies = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    start_time = time.time()
                    outputs = model(inputs)
                    latencies.append(time.time() - start_time)
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Metric Calculations
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            avg_latency = np.mean(latencies) * 1000 # convert to ms
            class_f1 = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

            performance_data.append({
                'Architecture': arch_name,
                'Feature': feat_name,
                'Accuracy': acc,
                'Latency_ms': avg_latency,
                'F1_Weighted': weighted_f1,
                'Abaca_F1': class_f1[0],
                'Daratex_F1': class_f1[1],
                'Mixed_F1': class_f1[2]
            })

            # Save Confusion Matrix
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                        xticklabels=GLOBAL_CLASSES, yticklabels=GLOBAL_CLASSES)
            plt.title(f"Confusion Matrix: {arch_name}_{feat_name}")
            plt.savefig(f"cm_{arch_name}_{feat_name}.png")
            plt.close()


    df = pd.DataFrame(performance_data)
    
    # --- New Detailed Visualizations ---
    
    # 1. Performance vs Latency Scatter (The "Choice" Graph)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Latency_ms', y='Accuracy', hue='Architecture', style='Feature', s=100)
    plt.title("System Efficiency: Accuracy vs Inference Speed")
    plt.xlabel("Latency per Image (ms)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig("efficiency_tradeoff.png", dpi=300)

    # 2. Radar Chart for the 'All' feature
    plot_radar_chart(df[df['Feature'] == 'all'], GLOBAL_CLASSES)

    print("\n--- Evaluation Done ---")
    print(df[['Architecture', 'Feature', 'Accuracy', 'Latency_ms', 'F1_Weighted']])

DATA_ROOT = r"C:\Users\yen\Downloads\fiber-optics-backend-dev\services\uploads\labeled_dataset"
MODEL_DIR = r"./"

if __name__ == "__main__":
    # This calls the function and stores the results in a DataFrame
    summary_df = evaluate_all_models(DATA_ROOT, model_dir=MODEL_DIR)
    
    # Optional: Save the raw results to a CSV for your thesis records
    summary_df.to_csv("final_evaluation_results.csv", index=False)
    
    print("\n[SUCCESS] Evaluation complete. Check your folder for .png files and the LaTeX table.")