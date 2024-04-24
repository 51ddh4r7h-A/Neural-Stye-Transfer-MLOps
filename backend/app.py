"""FastAPI App."""

import sys
sys.path.insert(0, './src')

import boto3
import os, sys
import shutil
import tempfile
import torch, mlflow.pytorch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.models.network import StyleTransferNetwork
from src.utils.image_utils import imload, imsave
from src.configs import config
from botocore.exceptions import ClientError
import urllib.request


# Initialize FastAPI app
app = FastAPI()

mlflow.set_tracking_uri("https://dagshub.com/shatter-star/musical-octo-dollop.mlflow")

# Initialize CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize StyleTransferNetwork model
device = torch.device('cpu')
model_uri = "mlflow-artifacts:/366666ce4dc8413383fd5d9a1ce802f9/8c9c0df67b1d4151886eec4a77c36417/artifacts/model"
model = mlflow.pytorch.load_model(model_uri, map_location=device)
model.eval()

S3_BUCKET_NAME = 'my-style-bucket'
S3_STYLIZED_IMAGE_PREFIX = 'fused_images/'

# Define the endpoint for stylizing images
@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_index: int = 0):
    try:
        # Validate the style index
        if style_index != -1 and style_index not in range(config.NUM_STYLE):
            raise HTTPException(status_code=400, detail="Invalid style index")

        # Save uploaded content image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_content:
            shutil.copyfileobj(content_image.file, temp_content)
            content_path = temp_content.name

        # Generate filename for the stylized image
        content_filename, content_extension = os.path.splitext(content_image.filename)
        output_filename = f"stylized_{content_filename}{content_extension}"
        output_path = os.path.join('/tmp', output_filename)

        # Load content image and apply style transfer
        content_image = imload(content_path, imsize=config.IMSIZE)
        if style_index == -1:
            style_code = torch.eye(config.NUM_STYLE).unsqueeze(-1)
            content_image = content_image.repeat(config.NUM_STYLE, 1, 1, 1)
        else:
            style_code = torch.zeros(1, config.NUM_STYLE, 1)
            style_code[:, style_index, :] = 1
        stylized_image = model(content_image, style_code)
        imsave(stylized_image, output_path)

        # Generate the URL for the stylized image
        s3_key = f"{S3_STYLIZED_IMAGE_PREFIX}{output_filename}"
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

        # Return the stylized image as a streaming response
        file_like = open(output_path, mode="rb")
        return StreamingResponse(file_like, media_type='image/jpeg')

    finally:
        # Remove temporary content file and stylized image file
        os.unlink(content_path)
        os.unlink(output_path)

@app.get("/")
def root():
    return {"message": "Style Transfer API is running!"}