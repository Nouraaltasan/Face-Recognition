import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="Face Detection & Captioning", layout="centered")
st.title("🖼️ Face Detection & Image Captioning")

# Load captioning model
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()

# Captioning function
def generate_caption(image_pil):
    inputs = processor(images=image_pil, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Face detection function
def detect_faces(image_np):
    face_locations = face_recognition.face_locations(image_np)
    for top, right, bottom, left in face_locations:
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
    return image_np

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    st.subheader("Original Image")
    st.image(image_pil, use_column_width=True)

    with st.spinner("Detecting faces..."):
        image_with_faces = detect_faces(image_np.copy())
    st.subheader("Image with Faces")
    st.image(image_with_faces, use_column_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image_pil)
    st.subheader("Generated Caption")
    st.markdown(f"**{caption}**")

