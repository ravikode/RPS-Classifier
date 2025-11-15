import tensorflow as tf
from pathlib import Path
import datetime
import re
from google.cloud import storage

# --- Configuration ---
GCS_BUCKET_NAME = "ravikode-rps-project-data"
GCS_MODEL_ROOT_DIR = "models" 

# Local paths (on the GCP cloud computer)
OUTPUT_DIR = Path("./RPS_Output")
SAVED_MODEL_DIR = str(OUTPUT_DIR / "RPS_model")
# --- THIS PATH IS UPDATED ---
# Point to the LOCAL dataset folder
DATA_DIR = "./local_dataset" 
# ----------------------------
TFLITE_SAVE_DIR = OUTPUT_DIR 
IMG_SIZE = (224, 224)
# ---------------------

TFLITE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
fp16_success, int8_success = False, False

# (FP16 conversion is unchanged)
print("Starting FP16 conversion...")
try:
    converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_model_fp16 = converter_fp16.convert()
    fp16_path = TFLITE_SAVE_DIR / "RPS_fp16.tflite"
    fp16_path.write_bytes(tflite_model_fp16)
    print(f"FP16 model saved to: {fp16_path} ({fp16_path.stat().st_size / (1024 * 1024):.2f} MB)")
    fp16_success = True
except Exception as e: 
    print(f"Error during FP16 conversion: {e}")


# --- 2. INT8 (Integer 8-bit) Conversion ---
print("\nStarting INT8 conversion...")

def representative_data_gen():
    print("  Loading representative dataset from LOCAL folder...")
    # --- THIS PATH IS UPDATED ---
    # Load from the local 'train' folder
    train_ds = tf.keras.utils.image_dataset_from_directory(str(Path(DATA_DIR)/"train"), image_size=IMG_SIZE, batch_size=1).take(150)
    # ----------------------------
    print("  Starting calibration...")
    for images, _ in train_ds: yield [images]
    print("  Calibration finished.")

try:
    # (INT8 conversion logic is unchanged)
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32 
    converter_int8.inference_output_type = tf.float32
    tflite_model_int8 = converter_int8.convert()
    int8_path = TFLITE_SAVE_DIR / "RPS_int8.tflite"
    int8_path.write_bytes(tflite_model_int8)
    print(f"INT8 model saved to: {int8_path} ({int8_path.stat().st_size / (1024 * 1024):.2f} MB)")
    int8_success = True
except Exception as e: 
    print(f"\nError during INT8 conversion: {e}")


# (GCS Upload logic is unchanged)
print(f"\n--- Conversion Summary ---")
print(f"{'✅' if fp16_success else '❌'} FP16 conversion")
print(f"{'✅' if int8_success else '❌'} INT8 conversion")

if int8_success:
    print("\nStarting model versioning and GCS upload...")
    
    today_str = datetime.datetime.now().strftime('%Y.%m.%d')
    folder_prefix = f"{GCS_MODEL_ROOT_DIR}/{today_str}/"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    print(f"Checking for existing builds in: {GCS_BUCKET_NAME}/{folder_prefix}...")
    existing_blobs = bucket.list_blobs(prefix=folder_prefix)
    
    build_num_regex = re.compile(r"RPS_int8_[\d\.]+_(\d+)\.tflite")
    
    max_build_num = 0
    for blob in existing_blobs:
        match = build_num_regex.search(blob.name)
        if match:
            found_num = int(match.group(1))
            if found_num > max_build_num:
                max_build_num = found_num
                
    new_build_num = max_build_num + 1
    print(f"Highest build found: {max_build_num}. Setting new build number to: {new_build_num}")

    versioned_name = f"RPS_int8_{today_str}_{new_build_num}.tflite"
    
    local_model_path = OUTPUT_DIR / "RPS_int8.tflite"
    local_label_path = OUTPUT_DIR / "labels.txt"
    
    gcs_model_path = f"{folder_prefix}{versioned_name}"
    gcs_label_path = f"{folder_prefix}labels.txt"

    model_blob = bucket.blob(gcs_model_path)
    model_blob.upload_from_filename(local_model_path)
    print(f"✅ Successfully uploaded model to: {gcs_model_path}")
    
    label_blob = bucket.blob(gcs_label_path)
    label_blob.upload_from_filename(local_label_path)
    print(f"✅ Successfully uploaded labels to: {gcs_label_path}")
    
    print("GCS upload complete.")
else:
    print("❌ INT8 conversion failed. Skipping GCS upload.")
