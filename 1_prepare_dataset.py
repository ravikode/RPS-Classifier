import os, shutil, random
from pathlib import Path

# --- Configuration ---
# Use relative paths to work on any computer
SOURCE_DIR = "../rps-dataset"
DEST_DATASET_DIR = "../dataset"
# ---------------------

def prepare_dataset(source_dir, dest_base_dir, split_ratios=(0.8, 0.1, 0.1)):
    source_path = Path(source_dir)
    dest_path = Path(dest_base_dir)
    
    if dest_path.exists():
        print(f"Destination directory {dest_path} already exists. Removing it.")
        shutil.rmtree(dest_path)
        
    train_path = dest_path / "train"
    val_path = dest_path / "val"
    test_path = dest_path / "test"
    
    print(f"Starting dataset preparation from {source_path}")
    original_folders = [f for f in source_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    if not original_folders:
        print(f"Error: No subdirectories (rock, paper, scissors) found in {source_path}.")
        return

    for src_class_dir in original_folders:
        class_name = src_class_dir.name
        print(f"Processing '{class_name}'...")
        
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (val_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        extensions = ('.jpg', '.jpeg', '.png')
        images = [f for f in src_class_dir.rglob('*') if f.suffix.lower() in extensions]
        random.shuffle(images)
        
        total_images = len(images)
        train_split = int(total_images * split_ratios[0])
        val_split = int(total_images * (split_ratios[0] + split_ratios[1]))
        
        train_files = images[:train_split]
        val_files = images[train_split:val_split]
        test_files = images[val_split:]
        
        def copy_files(files, dest_split_path):
            for f in files: shutil.copy(f, dest_split_path / class_name / f.name)

        copy_files(train_files, train_path)
        copy_files(val_files, val_path)
        copy_files(test_files, test_path)
        
        print(f"  Total: {total_images} | Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    print(f"\nDataset preparation complete. Data is in: {dest_path}")

if __name__ == "__main__":
    prepare_dataset(SOURCE_DIR, DEST_DATASET_DIR)
