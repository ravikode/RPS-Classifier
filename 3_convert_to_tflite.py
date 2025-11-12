# --------------------------------------------------
# SCRIPT 3: 3_convert_to_tflite.py (FINAL)
# Converts model and uploads versioned file to GCS
# --------------------------------------------------

# Import TensorFlow for conversion
import tensorflow as tf
# Import Path for local file path handling
from pathlib import Path
# Import datetime to get the current date for versioning
import datetime
# Import re to parse filenames using regular expressions
import re
# Import the Google Cloud Storage client library
from google.cloud import storage

# --- Configuration ---
# Your GCS bucket name
GCS_BUCKET_NAME = "ravikode-rps-project-data"
# The top-level "folder" in your bucket for models
GCS_MODEL_ROOT_DIR = "models" 

# Local paths (on the GCP cloud computer)
OUTPUT_DIR = Path("./RPS_Output")
SAVED_MODEL_DIR = str(OUTPUT_DIR / "RPS_model")
DATA_DIR = "gs://ravikode-rps-project-data/Rock-Paper-Scissors" 
TFLITE_SAVE_DIR = OUTPUT_DIR 
IMG_SIZE = (224, 224)
# ---------------------

# Create the local output directory if it doesn't exist.
TFLITE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Initialize status flags.
fp16_success, int8_success = False, False

# --- 1. FP16 (Float16) Conversion ---
print("Starting FP16 conversion...")
try:
    # Initialize the TFLiteConverter from the SavedModel directory
    converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # Enable default optimizations
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set the target data type to float16
    converter_fp16.target_spec.supported_types = [tf.float16]
    # Run the conversion
    tflite_model_fp16 = converter_fp16.convert()
    # Define the output file path for the FP16 model
    fp16_path = TFLITE_SAVE_DIR / "RPS_fp16.tflite"
    # Write the converted model to a file
    fp16_path.write_bytes(tflite_model_fp16)
    # Print a success message with the path and file size
    print(f"FP16 model saved to: {fp16_path} ({fp16_path.stat().st_size / (1024 * 1024):.2f} MB)")
    # Set the success flag to True
    fp16_success = True
except Exception as e: 
    # Print an error message if it fails
    print(f"Error during FP16 conversion: {e}")


# --- 2. INT8 (Integer 8-bit) Conversion ---
print("\nStarting INT8 conversion...")

# Define a generator function to provide a "representative dataset" for calibration
def representative_data_gen():
    # Print a status message
    print("  Loading representative dataset from GCS...")
    # Load a small sample (150 images) from the GCS training set
    train_ds = tf.keras.utils.image_dataset_from_directory(str(Path(DATA_DIR)/"train"), image_size=IMG_SIZE, batch_size=1).take(150)
    # Print a status message
    print("  Starting calibration...")
    # Loop through the sample images
    for images, _ in train_ds: 
        # 'yield' one batch of images at a time to the converter
        yield [images]
    # Print a status message when done
    print("  Calibration finished.")

try:
    # Initialize the TFLiteConverter
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # Enable default optimizations
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    # Tell the converter to use our generator function for calibration
    converter_int8.representative_dataset = representative_data_gen
    # Force the converter to use only INT8 operations
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Keep the model's input type as float32
    converter_int8.inference_input_type = tf.float32 
    # Keep the model's output type as float32
    converter_int8.inference_output_type = tf.float32
    
    # Run the conversion and quantization process
    tflite_model_int8 = converter_int8.convert()

    # Define the output file path for the INT8 model
    int8_path = TFLITE_SAVE_DIR / "RPS_int8.tflite"
    # Write the converted model to a file
    int8_path.write_bytes(tflite_model_int8)
    
    # Print a success message with the path and file size
    print(f"INT8 model saved to: {int8_path} ({int8_path.stat().st_size / (1024 * 1024):.2f} MB)")
    # Set the success flag to True
    int8_success = True
except Exception as e: 
    # Print an error message if it fails
    print(f"\nError during INT8 conversion: {e}")


# --- 3. Final Summary & GCS UPLOAD (New Logic) ---
# Print a final summary header
print(f"\n--- Conversion Summary ---")
# Print the FP16 conversion status
print(f"{'✅' if fp16_success else '❌'} FP16 conversion")
# Print the INT8 conversion status
print(f"{'✅' if int8_success else '❌'} INT8 conversion")

# Only run the upload if the INT8 model was created successfully
if int8_success:
    # Print a status message
    print("\nStarting model versioning and GCS upload...")
    
    # --- 1. GET NEXT BUILD NUMBER ---
    
    # Get today's date in your requested format "YYYY.MM.DD"
    today_str = datetime.datetime.now().strftime('%Y.%m.%d')
    # Define the GCS "folder" prefix for today's date
    folder_prefix = f"{GCS_MODEL_ROOT_DIR}/{today_str}/"
    
    # Initialize the GCS client (it will auto-authenticate on Vertex AI)
    storage_client = storage.Client()
    # Get your specific bucket
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    # List all files ("blobs") that are already in today's folder
    print(f"Checking for existing builds in: {GCS_BUCKET_NAME}/{folder_prefix}...")
    existing_blobs = bucket.list_blobs(prefix=folder_prefix)
    
    # This regex will find the build number (e.g., the "1" in "..._1.tflite")
    build_num_regex = re.compile(r"RPS_int8_[\d\.]+_(\d+)\.tflite")
    
    # Start the counter for the highest build number found
    max_build_num = 0
    # Loop through all files found in the folder
    for blob in existing_blobs:
        # Check if the filename matches our regex
        match = build_num_regex.search(blob.name)
        # If it matches...
        if match:
            # ...get the build number (as an integer)
            found_num = int(match.group(1))
            # ...and check if it's the highest one found so far
            if found_num > max_build_num:
                # If it is, update our counter
                max_build_num = found_num
                
    # The new build number is the highest one found, plus 1
    new_build_num = max_build_num + 1
    # Print the result
    print(f"Highest build found: {max_build_num}. Setting new build number to: {new_build_num}")

    # --- 2. DEFINE FILE PATHS ---
    
    # Create the new versioned filename (e.g., "RPS_int8_2025.11.12_1.tflite")
    versioned_name = f"RPS_int8_{today_str}_{new_build_num}.tflite"
    
    # Get the paths to the local files we just created
    local_model_path = OUTPUT_DIR / "RPS_int8.tflite"
    local_label_path = OUTPUT_DIR / "labels.txt"
    
    # Define the full GCS destination path for the model
    gcs_model_path = f"{folder_prefix}{versioned_name}"
    # Define the GCS destination path for the labels
    gcs_label_path = f"{folder_prefix}labels.txt"

    # --- 3. UPLOAD FILES TO GCS ---
    
    # Get a "blob" object for the model's destination
    model_blob = bucket.blob(gcs_model_path)
    # Upload the local model file to that destination
    model_blob.upload_from_filename(local_model_path)
    # Print a success message
    print(f"✅ Successfully uploaded model to: {gcs_model_path}")
    
    # Get a "blob" object for the label's destination
    label_blob = bucket.blob(gcs_label_path)
    # Upload the local labels file
    label_blob.upload_from_filename(local_label_path)
    # Print a success message
    print(f"✅ Successfully uploaded labels to: {gcs_label_path}")
    
    # Print a final "all done" message
    print("GCS upload complete.")
else:
    # Print a failure message if the INT8 conversion failed
    print("❌ INT8 conversion failed. Skipping GCS upload.")
# --- END OF NEW SECTION ---
