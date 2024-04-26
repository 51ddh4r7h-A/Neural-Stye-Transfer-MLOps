"""Evaluation Script."""

import sys
import os
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

import argparse
import torch
import os
import mlflow
from mlflow.tracking import MlflowClient

from src.configs import config
from src.models import StyleTransferNetwork
from src.utils.image_utils import *
from src.utils.data_utils import *


def get_latest_model_uri():
    """Retrieve the URI of the latest MLflow model."""
    # Set up MLflow client
    mlflow.set_tracking_uri("https://dagshub.com/shatter-star/musical-octo-dollop.mlflow")
    client = MlflowClient()

    # Get the latest run
    latest_run = client.search_runs(experiment_ids="0", order_by=["attribute.start_time DESC"], max_results=1)[0]

    # Get the model URI from the latest run
    model_uri = latest_run.info.artifact_uri + "/model"

    return model_uri


def evaluate(args):
    """Evaluate the network."""
    device = torch.device('cpu')
    # Get the latest model URI
    model_uri = get_latest_model_uri()
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
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
    parser.add_argument('--model_uri', type=str, default=None,
                        help='URI of the MLflow model to load. If not specified, the latest model will be fetched.')

    args = parser.parse_args()

    if not args.model_uri:
        # Get the latest model URI if not provided
        args.model_uri = get_latest_model_uri()

    evaluate(args)
