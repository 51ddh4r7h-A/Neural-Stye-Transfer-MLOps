import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Define the FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/stylize/"

# Streamlit app title
st.title("Image Stylization with FastAPI and Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Select style index
style_index = st.slider("Select Style Index:", min_value=-1, max_value=15, value=0)

# Stylize image when 'Stylize' button is clicked
if st.button("Stylize"):
    if uploaded_file is not None:
        # Send image and style index to FastAPI endpoint
        files = {"content_image": uploaded_file}
        params = {"style_index": style_index}
        response = requests.post(FASTAPI_URL, files=files, params=params)

        # Display stylized image
        if response.status_code == 200:
            stylized_image = Image.open(BytesIO(response.content))
            st.image(stylized_image, caption="Stylized Image", use_column_width=True)
        else:
            st.error("Error occurred during stylization. Please try again.")
    else:
        st.warning("Please upload an image first.")
