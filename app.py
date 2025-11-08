import os
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np

# Load model
MODEL_PATH = "./model/digit_predict.keras"
model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a handwritten digit (0-9). If not 28x28, it'll be resized automatically.")

uploaded_file = st.file_uploader("📩 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if img_array.shape[:2] != (28, 28):
        img_array = cv2.resize(img_array, (28, 28))

    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    img_array = np.invert(img_array)

    img_input = img_array / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)

    prediction = model.predict(img_input)
    predicted_label = np.argmax(prediction)

    with col2:
        st.markdown("### 🔍 Model Prediction")
        st.markdown(f"<h1 style='text-alignment:center; color:#4CAF50;'>{predicted_label}</h1>", unsafe_allow_html=True)
        st.bar_chart(prediction[0])

else:
    st.info("👆 Upload a handwritten digit image to begin.")

st.markdown("---")
st.caption("Built with ❤️ by Saptarshi")
