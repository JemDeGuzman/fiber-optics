import os

def check_data_integrity(root_dir):
    categories = ['Abaca', 'Daratex', 'False']
    # If you have a 'Mixed' folder, add it here
    
    total_samples = 0
    
    print(f"{'Location':<40} | {'Category':<10} | {'Samples'}")
    print("-" * 65)

    # 1. Check Original/Root folders (should be empty if move was successful)
    for cat in categories:
        path = os.path.join(root_dir, cat, 'sample')
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])
            total_samples += count
            print(f"{cat + ' (Original leftovers)':<40} | {cat:<10} | {count}")

    # 2. Check Train folders
    for cat in categories:
        path = os.path.join(root_dir, 'train', cat, 'sample')
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])
            total_samples += count
            print(f"{'train/' + cat:<40} | {cat:<10} | {count}")

    # 3. Check Val folders
    for cat in categories:
        path = os.path.join(root_dir, 'val', cat, 'sample')
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])
            total_samples += count
            print(f"{'val/' + cat:<40} | {cat:<10} | {count}")

    print("-" * 65)
    print(f"TOTAL UNIQUE CAPTURES FOUND: {total_samples}")
    print(f"TOTAL ESTIMATED FILES (Samples x 4): {total_samples * 4}")

# Execute
DATA_ROOT = r"C:\Users\yen\Downloads\fiber-optics-backend-dev\services\uploads\labeled_dataset"
check_data_integrity(DATA_ROOT)