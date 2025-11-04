import tensorflow as tf, os
from pathlib import Path

# --- Configuration ---
# Use relative paths
DATA_DIR = "../dataset"
OUTPUT_DIR = "./RPS_Output"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "RPS_model")
LABEL_SAVE_DIR = OUTPUT_DIR

# Model parameters
IMG_SIZE = (224, 224); BATCH_SIZE = 32; EPOCHS = 50; LEARNING_RATE = 0.0001
# ---------------------

def build_model(num_classes):
    IMG_SHAPE = IMG_SIZE + (3,)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SHAPE, name="input_layer")
    x = preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")(x)
    return tf.keras.Model(inputs, outputs)

def main():
    train_dir, val_dir, test_dir = Path(DATA_DIR)/"train", Path(DATA_DIR)/"val", Path(DATA_DIR)/"test"
    
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    Path(LABEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    label_path = Path(LABEL_SAVE_DIR) / "labels.txt"
    print(f"Saving labels to {label_path}")
    with open(label_path, 'w') as f:
        for name in class_names: f.write(f"{name}\n")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ], name="augmentation_layer")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("Building model...")
    model = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\nStarting model training...")
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stopping])

    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(f"Saving model to {MODEL_SAVE_DIR}")
    Path(MODEL_SAVE_DIR).parent.mkdir(parents=True, exist_ok=True)
    model.export(MODEL_SAVE_DIR)
    print("Training complete.")

if __name__ == "__main__":
    main()
