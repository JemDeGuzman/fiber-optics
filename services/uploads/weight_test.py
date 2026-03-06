import torch
import os

def inspect_model_indices(model_path):
    print(f"--- Inspecting: {os.path.basename(model_path)} ---")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 1. Check if class_to_idx was saved in the checkpoint
        if isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
            print(f"Found saved mapping: {checkpoint['class_to_idx']}")
        else:
            print("No explicit class_to_idx found. Checking weight signatures...")

        # 2. Extract the weights of the final layer
        # We look for 'fc.weight' (ResNet) or 'classifier.weight' (CAILN)
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
        
        fc_weight = None
        for key in state_dict.keys():
            if 'fc.weight' in key or 'classifier' in key and 'weight' in key:
                if state_dict[key].shape[0] == 3: # Ensure it's the 3-class layer
                    fc_weight = state_dict[key]
                    break
        
        if fc_weight is not None:
            # Calculate the 'Magnitude' of each class's feature detector
            # This tells us which class the model is most "sensitive" to
            magnitudes = torch.norm(fc_weight, dim=1)
            print(f"Class Sensitivities (Norms):")
            print(f"  Index 0: {magnitudes[0]:.4f}")
            print(f"  Index 1: {magnitudes[1]:.4f}")
            print(f"  Index 2: {magnitudes[2]:.4f}")
            
            if magnitudes[2] > magnitudes[0] and magnitudes[2] > magnitudes[1]:
                print("!! Index 2 (Mixed) has the strongest feature detectors. This usually causes over-classification !!")
        else:
            print("Could not locate the final classification layer.")
            
    except Exception as e:
        print(f"Error: {e}")

# Run it on your fusion model
inspect_model_indices("best_CNN_all_mixed.pth")