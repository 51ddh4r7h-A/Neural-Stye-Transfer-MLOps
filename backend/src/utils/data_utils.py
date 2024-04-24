"""Data Utility Code."""

import torch
from PIL import Image
from .image_utils import get_transforms


class ImageDataset:
    """Image Dataset."""

    def __init__(self, dir_path):
        """Init."""
        self.images = sorted(list(dir_path.glob('*.jpg')))

    def __len__(self):
        """Return the Number of data sampels."""
        return len(self.images)

    def __getitem__(self, index):
        """Get Image and Index."""
        img = Image.open(self.images[index]).convert('RGB')
        return img, index


class DataProcessor:
    """Data Processor."""

    def __init__(self, imsize=256, cropsize=240, cencrop=False):
        """Init."""
        self.transforms = get_transforms(imsize=imsize,
                                         cropsize=cropsize,
                                         cencrop=cencrop)

    def __call__(self, batch):
        """Process the batch."""
        images, indices = list(zip(*batch))

        inputs = torch.stack([self.transforms(image) for image in images])
        return inputs, indices