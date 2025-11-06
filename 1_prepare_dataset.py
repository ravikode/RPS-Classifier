# Import the 'os' module for basic operating system interactions.
import os
# Import 'shutil' for high-level file operations (copy, remove).
import shutil
# Import 'random' to shuffle the list of images for an unbiased split.
import random
# Import 'Path' from 'pathlib' for a modern, easy way to handle file paths.
from pathlib import Path

# --- Configuration ---
# Set the source directory to the 'rps-dataset' folder, which is one level *up* from this script.
SOURCE_DIR = "../rps-dataset"
# Set the destination for the new, organized dataset, also one level up.
DEST_DATASET_DIR = "../dataset"
# ---------------------

# Define the main function that splits the dataset.
def prepare_dataset(source_dir, dest_base_dir, split_ratios=(0.8, 0.1, 0.1)):
    # Convert the source directory string into a Path object.
    source_path = Path(source_dir)
    # Convert the destination directory string into a Path object.
    dest_path = Path(dest_base_dir)
    
    # Check if the destination folder (e.g., '../dataset') already exists.
    if dest_path.exists():
        # Print a message saying the old folder is being removed.
        print(f"Destination directory {dest_path} already exists. Removing it.")
        # Delete the entire folder and its contents to ensure a clean start.
        shutil.rmtree(dest_path)
        
    # Define the path for the 'train' subfolder.
    train_path = dest_path / "train"
    # Define the path for the 'val' (validation) subfolder.
    val_path = dest_path / "val"
    # Define the path for the 'test' subfolder.
    test_path = dest_path / "test"
    
    # Print a status message indicating where the script is reading from.
    print(f"Starting dataset preparation from {source_path}")
    # Get a list of all subfolders (e.g., 'rock', 'paper') in the source directory.
    # It ignores any hidden files (like .DS_Store).
    original_folders = [f for f in source_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    # If the list of folders is empty, print an error.
    if not original_folders:
        # This error means the 'rps-dataset' folder was not found or is empty.
        print(f"Error: No subdirectories (rock, paper, scissors) found in {source_path}.")
        # Stop the script.
        return

    # Loop through each class folder found (e.g., 'rock', 'paper', 'scissors').
    for src_class_dir in original_folders:
        # Get the name of the class from the folder's name.
        class_name = src_class_dir.name
        # Print the class being processed.
        print(f"Processing '{class_name}'...")
        
        # Create the destination subfolder for this class within 'train' (e.g., ../dataset/train/rock).
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        # Create the destination subfolder for this class within 'val'.
        (val_path / class_name).mkdir(parents=True, exist_ok=True)
        # Create the destination subfolder for this class within 'test'.
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Define the allowed image file extensions.
        extensions = ('.jpg', '.jpeg', '.png')
        # Get a list of all files in the class folder that match the extensions.
        images = [f for f in src_class_dir.rglob('*') if f.suffix.lower() in extensions]
        # Randomly shuffle the order of images for an unbiased split.
        random.shuffle(images)
        
        # Count the total number of images found for this class.
        total_images = len(images)
        # Calculate the split point for the training set (80%).
        train_split = int(total_images * split_ratios[0])
        # Calculate the split point for the validation set (80% + 10% = 90%).
        val_split = int(total_images * (split_ratios[0] + split_ratios[1]))
        
        # Select the first 80% of images for training.
        train_files = images[:train_split]
        # Select the next 10% for validation.
        val_files = images[train_split:val_split]
        # Select the final 10% for testing.
        test_files = images[val_split:]
        
        # Define a helper function to copy files.
        def copy_files(files, dest_split_path):
            # Loop through each file in the list.
            for f in files: 
                # Copy the file to its new home (e.g., .../train/rock/image_name.png).
                shutil.copy(f, dest_split_path / class_name / f.name)

        # Copy the training files to the 'train' folder.
        copy_files(train_files, train_path)
        # Copy the validation files to the 'val' folder.
        copy_files(val_files, val_path)
        # Copy the test files to the 'test' folder.
        copy_files(test_files, test_path)
        
        # Print a summary for this class.
        print(f"  Total: {total_images} | Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Print a final message when all classes are done.
    print(f"\nDataset preparation complete. Data is in: {dest_path}")

# This ensures the script only runs when executed directly.
if __name__ == "__main__":
    # Call the main function with the configured paths.
    prepare_dataset(SOURCE_DIR, DEST_DATASET_DIR)
