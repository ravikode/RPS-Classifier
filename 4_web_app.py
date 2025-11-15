import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Configuration ---
# !!! IMPORTANT !!!
# You must MANUALLY update this path to match the versioned model
# you downloaded from GCS.
MODEL_PATH = "RPS_Output/RPS_int8_2025.11.15_1.tflite" # <-- CHANGE THIS TO YOUR FILENAME
# ---------------------

LABEL_PATH = "RPS_Output/labels.txt"
MODEL_INPUT_SIZE = (224, 224)

@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        with open(LABEL_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return interpreter, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Make sure the file exists at: {MODEL_PATH}")
        return None, None

interpreter, labels = load_model()

# --- Web Page UI ---
st.set_page_config(layout="wide")
st.title("Ravikiran's Game Page")
st.write("Using the trained model, create a web application with the following specifications:")
st.markdown("""
- A single-page interface containing an **"Image Upload"** button.
- The user uploads an image of a hand showing **Rock, Paper, or Scissors**.
- The website processes the image using the trained model and outputs a message indicating the detected gesture.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    image_resized = image.resize(MODEL_INPUT_SIZE)
    input_tensor = np.expand_dims(np.array(image_resized, dtype=np.float32), axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output_data)
    
    best_index = np.argmax(predictions)
    label = labels[best_index]
    score = predictions[best_index]
    
    st.success(f"**Prediction: {label}**")
    st.info(f"**Confidence:** {int(score * 100)}%")
