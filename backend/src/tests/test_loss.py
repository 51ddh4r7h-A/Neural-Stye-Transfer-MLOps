import sys
import os
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

import unittest
import torch
from models.loss import calc_content_loss, gram, calc_style_loss, calc_tv_loss

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.features = {'node1': torch.randn(1, 3, 64, 64), 'node2': torch.randn(1, 3, 64, 64)}
        self.targets = {'node1': torch.randn(1, 3, 64, 64), 'node2': torch.randn(1, 3, 64, 64)}
        self.nodes = ['node1', 'node2']
        self.x = torch.randn(1, 3, 64, 64)

    def test_calc_content_loss(self):
        loss = calc_content_loss(self.features, self.targets, self.nodes)
        self.assertIsInstance(loss, torch.Tensor)

    def test_gram(self):
        g = gram(self.x)
        self.assertEqual(g.shape, (1, 3, 3))

    def test_calc_style_loss(self):
        loss = calc_style_loss(self.features, self.targets, self.nodes)
        self.assertIsInstance(loss, torch.Tensor)

    def test_calc_tv_loss(self):
        loss = calc_tv_loss(self.x)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()