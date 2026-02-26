import os
import pandas as pd
import shutil
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
csv_images = 'db_export.csv'
csv_samples = 'db_export_2.csv'
source_folder = '../uploads'
output_base = './labeled_dataset'
TRAIN_RATIO = 0.8 

# Extensions that are safe to delete during cleanup
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def organize_split_and_protected_cleanup():
    # Load Data
    if not os.path.exists(csv_images) or not os.path.exists(csv_samples):
        print("Error: CSV files missing.")
        return

    df_images = pd.read_csv(csv_images, header=None)
    df_samples = pd.read_csv(csv_samples, header=None)

    # Column Mapping
    df_images = df_images.rename(columns={1: 'sampleId', 2: 'url', 4: 'target_name'})
    df_samples = df_samples.rename(columns={0: 'id', 3: 'classification'})

    # --- 2. Perform the Split ---
    unique_ids = df_samples['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, train_size=TRAIN_RATIO, random_state=42)
    
    split_map = {id: 'train' for id in train_ids}
    split_map.update({id: 'val' for id in val_ids})

    merged_df = pd.merge(df_images, df_samples, left_on='sampleId', right_on='id')

    print(f"Total logical samples: {len(unique_ids)}")
    print(f"Processing {len(merged_df)} files...")

    success_count = 0
    missing_count = 0

    # --- 3. Process and MOVE ---
    for _, row in merged_df.iterrows():
        try:
            sample_id = row['sampleId']
            split_folder = split_map.get(sample_id, 'val') 
            classification = str(row['classification']).strip()
            
            if not classification or classification == 'nan':
                classification = "Unlabeled"

            img_type = str(row['target_name']).split('-')[0]
            original_filename = os.path.basename(urlparse(str(row['url'])).path)
            
            target_dir = os.path.join(output_base, split_folder, classification, img_type)
            os.makedirs(target_dir, exist_ok=True)

            source_path = os.path.join(source_folder, original_filename)
            target_path = os.path.join(target_dir, str(row['target_name']))

            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                success_count += 1
            else:
                missing_count += 1
                
        except Exception as e:
            print(f"Error at sample {row.get('sampleId')}: {e}")

    # --- 4. Protected Cleanup ---
    print("\nStarting PROTECTED cleanup (images only)...")
    remaining_files = os.listdir(source_folder)
    deleted_leftovers = 0
    skipped_scripts = 0
    
    for f in remaining_files:
        file_path = os.path.join(source_folder, f)
        if os.path.isfile(file_path):
            # Only delete if it's an image file
            if f.lower().endswith(IMAGE_EXTENSIONS):
                try:
                    os.remove(file_path)
                    deleted_leftovers += 1
                except Exception as e:
                    print(f"Could not delete {f}: {e}")
            else:
                # Log that we are skipping non-image files (like .py or .js)
                skipped_scripts += 1

    print("-" * 30)
    print(f"Process Complete!")
    print(f"Successfully moved: {success_count} files")
    print(f"Unlabeled images deleted: {deleted_leftovers}")
    print(f"Non-image files preserved: {skipped_scripts}")

if __name__ == "__main__":
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    organize_split_and_protected_cleanup()