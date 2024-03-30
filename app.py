from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import tempfile
import os
import torch
import mlflow.pytorch
from models import StyleTransferNetwork
from utils.image_utils import imload, imsave
from configs import config

# Set up MLflow with provided environment variables
mlflow.set_tracking_uri("https://dagshub.com/shatter-star/musical-octo-dollop.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "shatter-star"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "411996890a0df0c0ccf65dbd848d454f40ad3cbb"

# Initialize FastAPI app
app = FastAPI()

# Initialize StyleTransferNetwork model
device = torch.device('cpu')
model_uri = "mlflow-artifacts:/366666ce4dc8413383fd5d9a1ce802f9/d3b22973d11e4765a60d82a68edca4d7/artifacts/model"
model = mlflow.pytorch.load_model(model_uri, map_location=device)
model.eval()

# Create a directory for storing stylized images
STYLIZED_IMAGE_DIR = "stylized_images"
os.makedirs(STYLIZED_IMAGE_DIR, exist_ok=True)

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
        output_path = os.path.join(STYLIZED_IMAGE_DIR, output_filename)

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

        # Return the stylized image file
        return FileResponse(output_path, media_type='image/jpeg', filename=output_filename)
    
    finally:
        # Remove temporary content file
        os.unlink(content_path)


@app.get("/")
def root():
    return {"message": "Style Transfer API is running!"}
