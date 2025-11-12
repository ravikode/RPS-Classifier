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

# --- Configuration ---
# !!! IMPORTANT !!!
# You must MANUALLY update this path to match the versioned model
# you downloaded from GCS.
MODEL_PATH = "RPS_Output/RPS_int8_2025.11.12_1.tflite" # <-- CHANGE THIS
# ---------------------

# Define the path to the labels file
LABEL_PATH = "RPS_Output/labels.txt"
# Define the target size for the model's input
MODEL_INPUT_SIZE = (224, 224)


# Function to load the model (cached so it only loads once)
@st.cache_resource
def load_model():
    # Use a try...except block to catch errors
    try:
        # Load the TFLite model and allocate memory for it
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
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
        st.error(f"Make sure the file exists at: {MODEL_PATH}")
        # Return nothing
        return None, None

# Call the load_model function to get the interpreter and labels
interpreter, labels = load_model()

# --- Web Page UI ---
# Set the page layout to "wide"
st.set_page_config(layout="wide")
# Set the title of the web page (as requested)
st.title("Ravikiran's Game Page")
# Write a short description
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
    # Write a status message
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
