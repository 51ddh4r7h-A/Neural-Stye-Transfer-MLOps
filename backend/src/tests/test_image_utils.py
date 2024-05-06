import sys
import os
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

import unittest
import torch
from pathlib import Path
from utils.image_utils import get_transforms, imload, imsave
import torchvision.transforms as T

class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self.image_path = Path('./imgs/gentlecat.png')
        self.save_path = Path('./stylized_image.png')

    def test_get_transforms(self):
        transformer = get_transforms(imsize=256, cropsize=240, cencrop=True)
        self.assertIsInstance(transformer, T.Compose)

    def test_imload(self):
        image = imload(self.image_path, imsize=256, cropsize=240, cencrop=True)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dim(), 4)  # Check if image has 4 dimensions (batch, channels, height, width)

    def test_imsave(self):
        image = imload(self.image_path, imsize=256, cropsize=240, cencrop=True)
        imsave(image, self.save_path.with_suffix('.jpg'))  # Save as .jpg
        self.assertTrue(self.save_path.with_suffix('.jpg').exists())  # Check if .jpg image was saved
        imsave(image, self.save_path.with_suffix('.png'))  # Save as .png
        self.assertTrue(self.save_path.with_suffix('.png').exists())  # Check if .png image was saved

if __name__ == '__main__':
    unittest.main()