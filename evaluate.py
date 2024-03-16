"""Evaluation Script."""

import argparse
import torch

from configs import config
from models import StyleTransferNetwork
from utils.image_utils import *
from utils.data_utils import *

def evaluate(args):
    """Evaluate the network."""
    device = torch.device('cpu')
    ckpt = torch.load(args.checkpoint_path, map_location=device)

    model = StyleTransferNetwork(num_style=config.NUM_STYLE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    content_image = imload(args.content_path, imsize=args.imsize)
    # for all styles
    if args.style_index == -1:
        style_code = torch.eye(config.NUM_STYLE).unsqueeze(-1)
        content_image = content_image.repeat(config.NUM_STYLE, 1, 1, 1)

    # for specific style
    elif args.style_index in range(config.NUM_STYLE):
        style_code = torch.zeros(1, config.NUM_STYLE, 1)
        style_code[:, args.style_index, :] = 1

    else:
        raise RuntimeError("Not expected style index")

    stylized_image = model(content_image, style_code)
    imsave(stylized_image, args.output_path)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data configurations
    parser.add_argument('--content_path', type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--imsize', type=int, default=config.IMSIZE,
                        help='Input image size')

    # Other configurations
    parser.add_argument('--checkpoint_path', type=str, default='model.ckpt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_path', type=str, default='stylized_image.jpg',
                        help='Path to save the stylized image')
    parser.add_argument('--style_index', type=int, default=0,
                        help='Index of the style to use (-1 for all styles)')

    args = parser.parse_args()

    evaluate(args)