# --------------------------------------------------
# SCRIPT 2: 2_train_model.py (FINAL)
# Reads data from GCS and trains the model
# --------------------------------------------------

# Import the TensorFlow library for machine learning
import tensorflow as tf
# Import the 'os' module to help build file paths
import os
# Import the 'Path' class from 'pathlib' for easy path manipulation
from pathlib import Path

# --- Configuration ---
# Point this to your GCS BUCKET path
DATA_DIR = "gs://ravikode-rps-project-data/Rock-Paper-Scissors"

# Define a new output directory (this will be on the cloud computer)
OUTPUT_DIR = "./RPS_Output"
# Create the full path for saving the trained model directory
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "RPS_model")
# Set the directory for saving the labels.txt file
LABEL_SAVE_DIR = OUTPUT_DIR

# Model parameters
IMG_SIZE = (224, 224); BATCH_SIZE = 32; EPOCHS = 50; LEARNING_RATE = 0.0001
# ---------------------

# Define the function that builds our neural network
def build_model(num_classes):
    # Define the full input shape (Height, Width, Color Channels)
    IMG_SHAPE = IMG_SIZE + (3,)
    # Get the specific preprocessing function required by the MobileNetV2 model
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Load the MobileNetV2 model, pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # Freeze the pre-trained layers
    base_model.trainable = False

    # Start building our new model "on top" of the base model
    inputs = tf.keras.Input(shape=IMG_SHAPE, name="input_layer")
    # Apply the MobileNetV2 preprocessing to the input images
    x = preprocess_input(inputs) 
    # Pass the preprocessed images through the frozen base model
    x = base_model(x, training=False)
    # Add a layer to average the features into a single 1D vector
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Add a dropout layer to prevent overfitting
    x = tf.keras.layers.Dropout(0.2)
    # Add our new, final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")(x)
    # Create the final model by defining its input and output layers
    return tf.keras.Model(inputs, outputs)

# Define the main function that runs the training pipeline
def main():
    # Define paths to the GCS folders
    train_dir = os.path.join(DATA_DIR, "train")
    # IMPORTANT: Your dataset uses the "validation" folder name
    val_dir = os.path.join(DATA_DIR, "validation") 
    test_dir = os.path.join(DATA_DIR, "test")
    
    # Print a status message to the console
    print(f"Loading datasets directly from GCS bucket: {DATA_DIR}")
    # Load the training dataset from the GCS directory
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # Load the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # Load the test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    # Get the class names (e.g., 'paper', 'rock', 'scissors') from the dataset
    class_names = train_ds.class_names
    # Count the total number of classes
    num_classes = len(class_names)
    # Print the classes found for confirmation
    print(f"Found {num_classes} classes: {class_names}")

    # Create the local output directory (e.g., './RPS_Output') if it doesn't exist
    Path(LABEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    # Define the full path for the 'labels.txt' file
    label_path = Path(LABEL_SAVE_DIR) / "labels.txt"
    # Print a status message
    print(f"Saving labels to {label_path}")
    # Open the labels file in "write" mode
    with open(label_path, 'w') as f:
        # Write each class name on its own line in the file
        for name in class_names: f.write(f"{name}\n")

    # Define the augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ], name="augmentation_layer")

    # 'AUTOTUNE' tells TensorFlow to automatically optimize performance settings
    AUTOTUNE = tf.data.AUTOTUNE
    # Apply augmentations *only* to the training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # We don't augment the validation or test data
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Print a status message
    print("Building model...")
    # Call our 'build_model' function to create the compiled model
    model = build_model(num_classes)
    # Configure the model for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Define an 'EarlyStopping' callback to stop training if it's not improving
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Print a status message
    print("\nStarting model training...")
    # Start the training process
    model.fit(train_ds,
              epochs=EPOCHS,
              validation_data=val_ds,
              callbacks=[early_stopping])

    # Print a status message
    print("\nEvaluating model on test set...")
    # Evaluate the final model against the 'test' dataset
    loss, accuracy = model.evaluate(test_ds)
    # Print the final accuracy
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print a status message
    print(f"Saving model to {MODEL_SAVE_DIR}")
    # Ensure the parent directory (./RPS_Output) exists
    Path(MODEL_SAVE_DIR).parent.mkdir(parents=True, exist_ok=True)
    # Export the model in the SavedModel format (a directory)
    model.export(MODEL_SAVE_DIR)
    # Print a final "all done" message
    print("Training complete.")

# This check ensures that the 'main()' function is only called when the script is run directly
if __name__ == "__main__":
    # Call the 'main' function to start the entire process
    main()
