"""Unit tests for models."""

import unittest
import torch

from models import StyleTransferNetwork, calc_content_loss, calc_style_loss, calc_tv_loss


class TestModels(unittest.TestCase):
    def setUp(self):
        self.num_style = 16
        self.batch_size = 2
        self.channels = 3
        self.height = 32
        self.width = 32

        self.content_features = {
            'relu_3_3': torch.randn(self.batch_size, 256, self.height // 8, self.width // 8)
        }
        self.style_features = {
            'relu_1_2': torch.randn(self.batch_size, 64, self.height // 4, self.width // 4),
            'relu_2_2': torch.randn(self.batch_size, 128, self.height // 8, self.width // 8),
            'relu_3_3': torch.randn(self.batch_size, 256, self.height // 16, self.width // 16),
            'relu_4_2': torch.randn(self.batch_size, 512, self.height // 32, self.width // 32)
        }
        self.output_features = {
            'relu_1_2': torch.randn(self.batch_size, 64, self.height // 4, self.width // 4),
            'relu_2_2': torch.randn(self.batch_size, 128, self.height // 8, self.width // 8),
            'relu_3_3': torch.randn(self.batch_size, 256, self.height // 16, self.width // 16),
            'relu_4_2': torch.randn(self.batch_size, 512, self.height // 32, self.width // 32)
        }
        self.output_images = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_content_loss(self):
        content_nodes = ['relu_3_3']
        loss = calc_content_loss(self.output_features, self.content_features, content_nodes)
        self.assertIsInstance(loss, torch.Tensor)

    def test_style_loss(self):
        style_nodes = ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_2']
        loss = calc_style_loss(self.output_features, self.style_features, style_nodes)
        self.assertIsInstance(loss, torch.Tensor)

    def test_tv_loss(self):
        loss = calc_tv_loss(self.output_images)
        self.assertIsInstance(loss, torch.Tensor)

    def test_style_transfer_network(self):
        model = StyleTransferNetwork(num_style=self.num_style)
        content_input = torch.randn(self.batch_size, self.channels, self.height, self.width)
        style_codes = torch.rand(self.batch_size, self.num_style, 1)
        output = model(content_input, style_codes)
        self.assertEqual(output.shape, (self.batch_size, self.channels, self.height, self.width))


if __name__ == '__main__':
    unittest.main()