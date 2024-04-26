"""Evaluation Script."""

import sys
import os
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

import argparse, torch, os, mlflow

from src.configs import config
from src.models import StyleTransferNetwork
from src.utils.image_utils import *
from src.utils.data_utils import *


def evaluate(args):
    """Evaluate the network."""
    device = torch.device('cpu')
    # Set up MLflow
    mlflow.set_tracking_uri("https://dagshub.com/shatter-star/musical-octo-dollop.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "shatter-star"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "411996890a0df0c0ccf65dbd848d454f40ad3cbb"
    model = mlflow.pytorch.load_model(args.model_uri, map_location=device)
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
    parser.add_argument('--output_path', type=str, default='stylized_image.jpg',
                        help='Path to save the stylized image')
    parser.add_argument('--style_index', type=int, default=0,
                        help='Index of the style to use (-1 for all styles)')
    parser.add_argument('--model_uri', type=str, required=False,
                        help='URI of the MLflow model to load')

    args = parser.parse_args()

    if not args.model_uri:
        args.model_uri = input("Enter the MLflow model URI: ")

    evaluate(args)