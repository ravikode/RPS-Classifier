# Import the TensorFlow library.
import tensorflow as tf
# Import the 'Path' class from 'pathlib' for easy path manipulation.
from pathlib import Path

# --- Configuration ---
# Define the path to the output folder (where models and labels are).
OUTPUT_DIR = Path("./RPS_Output")
# Define the path to the SavedModel directory created by script 2.
SAVED_MODEL_DIR = str(OUTPUT_DIR / "RPS_model")
# Define the path to the split 'dataset' folder (needed for INT8 calibration).
DATA_DIR = "../dataset" 
# Set the TFLite save directory (same as the output directory).
TFLITE_SAVE_DIR = OUTPUT_DIR 
# Define the image size (must match the model's input).
IMG_SIZE = (224, 224)
# ---------------------

# Create the output directory if it doesn't exist.
TFLITE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Initialize status flags for the final summary.
fp16_success, int8_success = False, False

# --- 1. FP16 (Float16) Conversion ---
# Print a status message.
print("Starting FP16 conversion...")
# Use a try...except block to catch any conversion errors.
try:
    # Initialize the TFLiteConverter from the SavedModel directory.
    converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # Enable default optimizations (like quantization).
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set the target data type to float16.
    converter_fp16.target_spec.supported_types = [tf.float16]
    # Run the conversion.
    tflite_model_fp16 = converter_fp16.convert()
    # Define the output file path for the FP16 model.
    fp16_path = TFLITE_SAVE_DIR / "RPS_fp16.tflite"
    # Write the converted model to a file in binary mode.
    fp16_path.write_bytes(tflite_model_fp16)
    # Print a success message with the path and file size.
    print(f"FP16 model saved to: {fp16_path} ({fp16_path.stat().st_size / (1024 * 1024):.2f} MB)")
    # Set the success flag to True.
    fp16_success = True
# If an error occurs during the 'try' block...
except Exception as e: 
    # ...print the error message.
    print(f"Error during FP16 conversion: {e}")

# --- 2. INT8 (Integer 8-bit) Conversion ---
# Print a status message.
print("\nStarting INT8 conversion...")
# Define a generator function to provide a "representative dataset" for calibration.
def representative_data_gen():
    # Print a status message.
    print("  Loading representative dataset...")
    # Load a small sample (150 images) from the training set.
    train_ds = tf.keras.utils.image_dataset_from_directory(Path(DATA_DIR)/"train", image_size=IMG_SIZE, batch_size=1).take(150)
    # Print a status message.
    print("  Starting calibration...")
    # Loop through the sample images.
    for images, _ in train_ds: 
        # 'yield' one batch of images at a time to the converter.
        yield [images]
    # Print a status message when done.
    print("  Calibration finished.")

# Use a try...except block to catch any quantization errors.
try:
    # Initialize the TFLiteConverter from the SavedModel directory.
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # Enable default optimizations.
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set the converter to use our representative dataset generator.
    converter_int8.representative_dataset = representative_data_gen
    # Force the converter to use only INT8 operations (full integer quantization).
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Keep the model's input type as float32 (it will handle conversion internally).
    converter_int8.inference_input_type = tf.float32 
    # Keep the model's output type as float32 (it will de-quantize the results).
    converter_int8.inference_output_type = tf.float32
    # Run the conversion and quantization process.
    tflite_model_int8 = converter_int8.convert()
    # Define the output file path for the INT8 model.
    int8_path = TFLITE_SAVE_DIR / "RPS_int8.tflite"
    # Write the converted model to a file in binary mode.
    int8_path.write_bytes(tflite_model_int8)
    # Print a success message with the path and file size.
    print(f"INT8 model saved to: {int8_path} ({int8_path.stat().st_size / (1024 * 1024):.2f} MB)")
    # Set the success flag to True.
    int8_success = True
# If an error occurs during the 'try' block...
except Exception as e: 
    # ...print the error message.
    print(f"\nError during INT8 conversion: {e}")

# --- 3. Final Summary ---
# Print a final summary of what succeeded and what failed.
print(f"\n--- Conversion Summary ---")
# Print the FP16 conversion status.
print(f"{'✅' if fp16_success else '❌'} FP16 conversion")
# Print the INT8 conversion status.
print(f"{'✅' if int8_success else '❌'} INT8 conversion")
