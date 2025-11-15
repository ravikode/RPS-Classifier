# --------------------------------------------------
# SCRIPT 4: 4_web_app.py (FINAL)
# Runs the web application on your local Mac
# --------------------------------------------------

# Import the 'streamlit' library, which is used to build the web app
import streamlit as st
# Import 'tensorflow' to use the TFLite interpreter
import tensorflow as tf
# Import 'numpy' for numerical operations on the image array
import numpy as np
# Import 'Image' from 'PIL' (Pillow) to open and process the uploaded image
from PIL import Image
# Import 'os' and 'glob' to find the model file automatically
import os
import glob

# --- Configuration ---
OUTPUT_DIR = "RPS_Output"
LABEL_PATH = f"{OUTPUT_DIR}/labels.txt"
MODEL_INPUT_SIZE = (224, 224)
# ---------------------

# Function to find the latest TFLite model automatically
def find_latest_model():
    # Look for all .tflite files in the output directory
    files = glob.glob(f"{OUTPUT_DIR}/*.tflite")
    if not files:
        return None
    # Sort files by modification time (newest first)
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Function to load the model (cached so it only loads once)
@st.cache_resource
def load_model(model_path):
    # Use a try...except block to catch errors
    try:
        # Load the TFLite model and allocate memory for it
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Open the labels file and read all lines into a list
        with open(LABEL_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        # Return the loaded model and labels
        return interpreter, labels
    # If loading fails (e.g., file not found)
    except Exception as e:
        # ...show an error message on the web page
        st.error(f"Error loading model: {e}")
        return None, None

# --- Web Page UI ---
st.set_page_config(layout="wide")
st.title("Ravikiran's Game Page")

# --- Auto-Detect Model ---
model_path = find_latest_model()

if model_path:
    st.success(f"Loaded model: **{os.path.basename(model_path)}**")
    # Call the load_model function
    interpreter, labels = load_model(model_path)
else:
    st.error(f"No .tflite model found in {OUTPUT_DIR}. Please download it from GCS.")
    interpreter = None
    labels = None

st.write("Using the trained model, create a web application with the following specifications:")
st.markdown("""
- A single-page interface containing an **"Image Upload"** button.
- The user uploads an image of a hand showing **Rock, Paper, or Scissors**.
- The website processes the image using the trained model and outputs a message indicating the detected gesture.
""")

# Create the "Image Upload" button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded AND if the model loaded successfully
if uploaded_file is not None and interpreter is not None:
    
    # --- 1. Process the uploaded image ---
    # Open the uploaded file as an image and ensure it's in RGB format
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image on the web page
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    # Resize the image to the 224x224 size the model expects
    image_resized = image.resize(MODEL_INPUT_SIZE)
    # Convert the resized image into a NumPy array
    # Add a 'batch' dimension (from 224,224,3 to 1,224,224,3)
    # Convert the data type to float32
    input_tensor = np.expand_dims(np.array(image_resized, dtype=np.float32), axis=0)
    
    # --- 2. Run Inference ---
    # Get the details for the model's input tensor
    input_details = interpreter.get_input_details()
    # Get the details for the model's output tensor
    output_details = interpreter.get_output_details()
    
    # Set the image data as the model's input
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    # Run the model
    interpreter.invoke()
    
    # Get the prediction results from the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Remove the unneeded 'batch' dimension from the results
    predictions = np.squeeze(output_data)
    
    # --- 3. Show Results ---
    # Find the index of the highest probability (the best guess)
    best_index = np.argmax(predictions)
    # Get the corresponding label name from our list
    label = labels[best_index]
    # Get the confidence score for that label
    score = predictions[best_index]
    
    # Display the final prediction in a green "success" box
    st.success(f"**Prediction: {label}**")
    # Display the confidence score in a blue "info" box
    st.info(f"**Confidence:** {int(score * 100)}%")
