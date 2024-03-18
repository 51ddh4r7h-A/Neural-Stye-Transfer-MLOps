import os
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import torch
from PIL import Image
import io

from configs import config
from models import StyleTransferNetwork
from utils.image_utils import imload, imsave

app = FastAPI()

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path(config.MODEL_PATH)
ckpt = torch.load(str(model_path), map_location=device)
model = StyleTransferNetwork(num_style=config.NUM_STYLE)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model = model.to(device)

# Define the path to the temporary directory in your GitHub repo
temp_dir = Path("tmp")

@app.post("/transfer_style", response_model=dict)
async def transfer_style(content_image: UploadFile = File(...), style_index: int = 0):
    """
    Transfer the style of a given image to a content image.

    Args:
        content_image (UploadFile): The content image file.
        style_index (int): The index of the desired style (0~15 for a specific style, or -1 for all styles). Default is 0.

    Returns:
        The absolute path to the stylized image.
    """
    # Ensure the temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded content image to a temporary file in the tmp folder
    content_image_name, content_image_extension = os.path.splitext(content_image.filename)
    stylized_image_name = f"stylized_{content_image_name}{content_image_extension}"
    content_image_path = temp_dir / content_image.filename
    stylized_image_path = temp_dir / stylized_image_name
    with open(content_image_path, "wb") as buffer:
        buffer.write(await content_image.read())

    # Load the content image and preprocess it
    content_tensor = imload(str(content_image_path), config.IMSIZE, config.IMSIZE)
    content_tensor = content_tensor.to(device)

    # Generate the style code
    if style_index == -1:
        style_code = torch.eye(config.NUM_STYLE).unsqueeze(-1).to(device)
        content_tensor = content_tensor.repeat(config.NUM_STYLE, 1, 1, 1)
        stylized_image = model(content_tensor, style_code)
    elif style_index in range(config.NUM_STYLE):
        style_code = torch.zeros(1, config.NUM_STYLE, 1, device=device)
        style_code[:, style_index, :] = 1
        stylized_image = model(content_tensor, style_code)
    else:
        raise ValueError("Invalid style index. Should be -1 or between 0 and 15.")

    # Save the stylized image with the same extension as the content image in the tmp folder
    imsave(stylized_image, str(stylized_image_path))

    # Remove the temporary content image file
    os.remove(content_image_path)

    # Construct the absolute file path
    absolute_filepath = str(stylized_image_path)

    # Return the absolute path
    return {"file_path": absolute_filepath} 

@app.get("/")
def root():
    return {"message": "Style Transfer API is running!"}
