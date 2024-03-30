from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import shutil
import tempfile
import os
import torch
import mlflow.pytorch
from models import StyleTransferNetwork
from utils.image_utils import imload, imsave
from configs import config
import boto3
from botocore.exceptions import ClientError

# Set up MLflow with provided environment variables
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
# Initialize FastAPI app
app = FastAPI()

# Initialize StyleTransferNetwork model
device = torch.device('cpu')
model_uri = os.environ["MODEL_URI"]
model = mlflow.pytorch.load_model(model_uri, map_location=device)
model.eval()

# Configure S3 client using environment variables
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ""),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ""),
    region_name=os.environ.get('AWS_DEFAULT_REGION', "")
)
S3_BUCKET_NAME = 'stylizedgenimages'
S3_STYLIZED_IMAGE_PREFIX = 'stylized_images/'

# Define the endpoint for stylizing images
@app.post("/stylize/")
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

        # Upload the stylized image to S3
        s3_key = f"{S3_STYLIZED_IMAGE_PREFIX}{output_filename}"
        try:
            s3_client.upload_file(output_path, S3_BUCKET_NAME, s3_key)
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Error uploading image to S3: {e}")

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