# Importing required libraries
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Set page title and layout
st.set_page_config(page_title="Face & Caption AI", layout="centered")

# Display the app title and instructions using HTML
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📸 Face Detection & Image Captioning</h1>
    <p style='text-align: center; color: gray;'>Upload an image or take a picture with your webcam!</p>
    <hr style='border: 1px solid #f0f0f0;' />
""", unsafe_allow_html=True)

# Load the BLIP captioning model and processor (cached to avoid reloading)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load the models once
processor, model = load_model()

# Load OpenCV's pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and draw bounding boxes
def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # Convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Detect faces
    for (x, y, w, h) in faces:
        # Draw green rectangles around detected faces
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_np

# Function to generate image caption using BLIP
def generate_caption(image_pil):
    inputs = processor(images=image_pil, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Let the user choose input method: upload or webcam
st.markdown("### 📁 Choose an input method")
input_method = st.radio("Select input type:", ("Upload Image", "Use Webcam"))

# Variable to hold the input image
image_pil = None

# Handle uploaded image
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")

# Handle webcam capture
elif input_method == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image_pil = Image.open(camera_image).convert("RGB")

# If an image was provided, process it
if image_pil is not None:
    image_np = np.array(image_pil)  # Convert PIL image to NumPy array

    # Show original image
    st.markdown("### 🖼️ Original Image")
    st.image(image_pil, use_column_width=True, caption="Input Image")

    # Detect faces and show result
    with st.spinner("🟩 Detecting faces..."):
        image_with_faces = detect_faces(image_np.copy())

    st.markdown("### 🔍 Image with Detected Faces")
    st.image(image_with_faces, use_column_width=True, caption="Faces Highlighted")

    # Generate and show caption
    with st.spinner("✍️ Generating a caption..."):
        caption = generate_caption(image_pil)

    st.markdown("### 📝 Generated Caption")
    st.success(f"**{caption}**")

    # Footer message with a privacy note
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Made with ❤️ using Streamlit, OpenCV & HuggingFace<br>"
        "🔒 Don't worry, your images are never saved."
        "</div>",
        unsafe_allow_html=True
    )

# If no image is uploaded or captured, show a prompt
else:
    st.info("⬆️ Please upload an image or take a picture to get started.")
