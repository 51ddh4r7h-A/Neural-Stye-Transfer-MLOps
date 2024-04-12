import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from pathlib import Path
import os

# Define the FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/stylize"

# Set page configuration
st.set_page_config(
    page_title="Image Stylization",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("frontend/style.css")

# Streamlit app title and description
st.title("Neural Style Transfer")
st.markdown(
    """
    Transform your images with various artistic styles using our cutting-edge AI-powered stylization tool.
    """
    """
    Styles Available:
    """
)

st.image("imgs/all_styles.png")

# Initialize session state
if "value" not in st.session_state:
    st.session_state.value = "Neural Style Transfer"

# Sidebar
with st.sidebar:
    if st.button("Reload"):
        st.session_state.value = "Neural Style Transfer"
        st.rerun()
    st.subheader("Settings")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    style_index = st.slider("Select Style Index:", min_value=-1, max_value=15, value=0)
    st.markdown("---")
    st.write("About this app:")
    st.write("This Streamlit app utilizes a FastAPI backend to perform image stylization. Select an image and a style index to get started.")
    st.write("-1: for all styles")
    st.write("0-15: for individual styles")


if st.button("Stylize"):
    if uploaded_file is not None:
        # Send image and style index to FastAPI endpoint
        files = {"content_image": uploaded_file}
        params = {"style_index": style_index}
        response = requests.post(FASTAPI_URL, files=files, params=params)

        if response.status_code == 200:
            stylized_image = Image.open(BytesIO(response.content))
            
            # Display the original uploaded image
            st.subheader("Original Image")
            st.image(uploaded_file, use_column_width=True)
            
            # Display the stylized image
            st.subheader("Stylized Image")
            st.image(stylized_image, use_column_width=True)
        else:
            st.error("Error occurred during stylization. Please try again.")
    else:
        st.warning("Please upload an image first.")

# Footer
with st.container():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: left; text-decoration: none;">
        <a href="https://github.com/shatter-star/musical-octo-dollop" target="_blank" style="color: #F3EDF0; text-decoration: none;">
            <i class="fab fa-github fa-2x"></i> GitHub
        </a>
        </div>
        """,
        unsafe_allow_html=True
    )