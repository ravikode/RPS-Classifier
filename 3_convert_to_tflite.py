import tensorflow as tf
from pathlib import Path

# --- Configuration ---
# Use relative paths
OUTPUT_DIR = Path("./RPS_Output")
SAVED_MODEL_DIR = str(OUTPUT_DIR / "RPS_model")
DATA_DIR = "../dataset" # The SPLIT dataset
TFLITE_SAVE_DIR = OUTPUT_DIR 
IMG_SIZE = (224, 224)
# ---------------------

TFLITE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
fp16_success, int8_success = False, False

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
except Exception as e: print(f"Error during FP16 conversion: {e}")

print("\nStarting INT8 conversion...")
def representative_data_gen():
    print("  Loading representative dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(Path(DATA_DIR)/"train", image_size=IMG_SIZE, batch_size=1).take(150)
    print("  Starting calibration...")
    for images, _ in train_ds: yield [images]
    print("  Calibration finished.")

try:
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
except Exception as e: print(f"\nError during INT8 conversion: {e}")

print(f"\n--- Conversion Summary ---")
print(f"{'✅' if fp16_success else '❌'} FP16 conversion")
print(f"{'✅' if int8_success else '❌'} INT8 conversion")
