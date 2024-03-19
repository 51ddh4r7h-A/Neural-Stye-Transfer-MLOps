from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import tempfile
import os
import torch
from models import StyleTransferNetwork
from utils.image_utils import imload, imsave
from configs import config

# Initialize FastAPI app
app = FastAPI()

# Initialize StyleTransferNetwork model
device = torch.device('cpu')
ckpt = torch.load('modelv2.ckpt', map_location=device)
model = StyleTransferNetwork(num_style=config.NUM_STYLE)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Create a directory for storing stylized images
STYLIZED_IMAGE_DIR = "stylized_images"
os.makedirs(STYLIZED_IMAGE_DIR, exist_ok=True)

# Define the endpoint for stylizing images
@app.post("/stylize/")
async def stylize(content_image: UploadFile = File(...), style_index: int = 0):
    try:
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
        style_code = torch.eye(config.NUM_STYLE).unsqueeze(-1) if style_index == -1 else torch.zeros(1, config.NUM_STYLE, 1)
        if style_index != -1 and style_index in range(config.NUM_STYLE):
            style_code[:, style_index, :] = 1
        else:
            raise HTTPException(status_code=400, detail="Invalid style index")

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