import sys
import os
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

import unittest
from pathlib import Path
from utils.data_utils import ImageDataset, DataProcessor

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.dataset = ImageDataset(Path('./imgs/style'))
        self.processor = DataProcessor()

    def test_ImageDataset_len(self):
        self.assertEqual(len(self.dataset), 16)  # Assuming you have 1 images in the test directory

    def test_ImageDataset_getitem(self):
        img, index = self.dataset[0]
        self.assertEqual(index, 0)
        self.assertEqual(img.mode, 'RGB')

    def test_DataProcessor_call(self):
        batch = [self.dataset[i] for i in range(5)]  # Create a batch with 5 images
        inputs, indices = self.processor(batch)
        self.assertEqual(len(inputs), 5)
        self.assertEqual(indices, tuple(range(5)))

if __name__ == '__main__':
    unittest.main()