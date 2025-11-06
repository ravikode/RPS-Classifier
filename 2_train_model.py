# Import the TensorFlow library for machine learning.
import tensorflow as tf
# Import the 'os' module to help build file paths.
import os
# Import the 'Path' class from 'pathlib' for easy path manipulation.
from pathlib import Path

# --- Configuration ---
# Set the path to the 'dataset' folder that script 1 created.
DATA_DIR = "../dataset"
# Set the name of the folder where the final model will be saved.
OUTPUT_DIR = "./RPS_Output"
# Combine the output path and model name for the full save path.
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "RPS_model")
# Set the location to save the 'labels.txt' file.
LABEL_SAVE_DIR = OUTPUT_DIR

# --- Model Hyperparameters ---
# Set the target image size (MobileNetV2 works well with 224x224).
IMG_SIZE = (224, 224)
# Set the number of images to process in each batch.
BATCH_SIZE = 32
# Set the maximum number of training cycles.
EPOCHS = 50
# Set the learning rate for the optimizer.
LEARNING_RATE = 0.0001
# ---------------------

# Define the function that builds our neural network.
def build_model(num_classes):
    # Define the full input shape (height, width, color channels).
    IMG_SHAPE = IMG_SIZE + (3,)
    # Get the specific preprocessing function required for MobileNetV2.
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Load the MobileNetV2 model, pre-trained on ImageNet.
    # 'include_top=False' means we don't include its final classification layer.
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # Freeze the pre-trained layers so we don't change them during initial training.
    base_model.trainable = False

    # Start building our new model using the Keras Functional API.
    # Define the input layer, which expects images of shape IMG_SHAPE.
    inputs = tf.keras.Input(shape=IMG_SHAPE, name="input_layer")
    # Apply the MobileNetV2 preprocessing to the input images.
    x = preprocess_input(inputs) 
    # Pass the preprocessed images to the frozen base model.
    x = base_model(x, training=False)
    # Add a pooling layer to average the features into a single vector.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Add a dropout layer to prevent overfitting (randomly "turns off" 20% of neurons).
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add our new, final classification layer.
    # It has 'num_classes' (e.g., 3) outputs and a 'softmax' activation
    # to convert outputs to probabilities.
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")(x)
    # Create the final model by specifying its inputs and outputs.
    return tf.keras.Model(inputs, outputs)

# Define the main function that runs the training pipeline.
def main():
    # Define the paths to our new 'train', 'val', and 'test' folders.
    train_dir, val_dir, test_dir = Path(DATA_DIR)/"train", Path(DATA_DIR)/"val", Path(DATA_DIR)/"test"
    
    # Print a status message.
    print("Loading datasets...")
    # Load the training dataset from the directory.
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # Load the validation dataset from the directory.
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # Load the test dataset from the directory.
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    # Get the class names (e.g., 'rock', 'paper', 'scissors') from the folder structure.
    class_names = train_ds.class_names
    # Get the total number of classes.
    num_classes = len(class_names)
    # Print the classes found.
    print(f"Found {num_classes} classes: {class_names}")

    # Create the output directory if it doesn't exist.
    Path(LABEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    # Define the full path for the 'labels.txt' file.
    label_path = Path(LABEL_SAVE_DIR) / "labels.txt"
    # Print a status message.
    print(f"Saving labels to {label_path}")
    # Open the file in write mode.
    with open(label_path, 'w') as f:
        # Write each class name on a new line.
        for name in class_names: f.write(f"{name}\n")

    # Define the augmentation layers as a sequential model.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ], name="augmentation_layer")

    # 'AUTOTUNE' tells TensorFlow to automatically find the best performance settings.
    AUTOTUNE = tf.data.AUTOTUNE
    # Apply the augmentations only to the training dataset using .map().
    # 'training=True' ensures augmentations only run during training.
    # '.shuffle(1000)' randomizes the order of data.
    # '.prefetch()' loads the next batch of data while the current one is processing.
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # We don't augment the validation or test data, just prefetch it.
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Print a status message.
    print("Building model...")
    # Call our function to create the model.
    model = build_model(num_classes)
    # Compile the model, configuring it for training.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), # Use the Adam optimizer.
                  loss='sparse_categorical_crossentropy', # Use this loss function for integer-based labels.
                  metrics=['accuracy']) # Track accuracy during training.
    # Print a summary of the model's architecture.
    model.summary()

    # Define an 'EarlyStopping' callback.
    # This will stop training if the validation loss ('val_loss')
    # doesn't improve for 5 epochs ('patience=5').
    # 'restore_best_weights=True' ensures we keep the weights from the best-performing epoch.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Print a status message.
    print("\nStarting model training...")
    # Start the training process.
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stopping])

    # Print a status message.
    print("\nEvaluating model on test set...")
    # Evaluate the trained model on the unseen test data.
    loss, accuracy = model.evaluate(test_ds)
    # Print the final test accuracy.
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print a status message.
    print(f"Saving model to {MODEL_SAVE_DIR}")
    # Ensure the parent directory exists.
    Path(MODEL_SAVE_DIR).parent.mkdir(parents=True, exist_ok=True)
    # Export the model in the SavedModel format (a directory), which is needed for TFLite conversion.
    model.export(MODEL_SAVE_DIR)
    # Print a final confirmation.
    print("Training complete.")

# This ensures the script only runs when executed directly.
if __name__ == "__main__":
    # Call the main function to start the process.
    main()
