"""Training Script."""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient

import mlflow.pytorch

from pathlib import Path
from configs import config
from models import StyleTransferNetwork, calc_content_loss, calc_style_loss, calc_tv_loss
from utils.data_utils import ImageDataset, DataProcessor
from utils.image_utils import imsave

def plot_losses(losses, run_id):
    """Plot loss graphs and log them as artifacts."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses['content'], label='Content Loss')
    plt.plot(losses['style'], label='Style Loss')
    plt.plot(losses['tv'], label='TV Loss')
    plt.plot(losses['total'], label='Total Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses Over Training')
    plt.legend()

    # Save the plot as an image file
    plot_path = 'losses.png'
    plt.savefig(plot_path)

    # Log the plot as an artifact
    mlflow.log_artifact(plot_path, artifact_path='plots', run_id=run_id)

    plt.close()

def train(args):
    """Train Network."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set up MLflow
    mlflow.set_tracking_uri("https://dagshub.com/shatter-star/musical-octo-dollop.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "shatter-star"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "411996890a0df0c0ccf65dbd848d454f40ad3cbb"
    mlflow_client = MlflowClient()

    experiment_name = "StyleTransferExperiment"
    experiment = mlflow_client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow_client.create_experiment(experiment_name)

    # data
    content_dataset = ImageDataset(dir_path=Path(args.content_path))
    style_dataset = ImageDataset(dir_path=Path(args.style_path))

    data_processor = DataProcessor(imsize=args.imsize,
                                   cropsize=args.cropsize,
                                   cencrop=args.cencrop)
    content_dataloader = DataLoader(dataset=content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=data_processor)
    style_dataloader = DataLoader(dataset=style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=data_processor)

    # loss network
    vgg = vgg16(pretrained=True).features  # Load with ImageNet weights
    for param in vgg.parameters():
        param.requires_grad = False
    loss_network = create_feature_extractor(vgg, config.RETURN_NODES).to(device)

    # network
    model = StyleTransferNetwork(num_style=config.NUM_STYLE)
    model.train()
    model = model.to(device)

    # optimizer# Use DataParallel to leverage multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    losses = {'content': [], 'style': [], 'tv': [], 'total': []}
    print("Start training...")

    with mlflow.start_run(experiment_id=experiment_id, run_name="StyleTransferRun") as run:
        run_id = run.info.run_id
        # Log parameters
        mlflow.log_params({
            "style_weight": args.style_weight,
            "tv_weight": args.tv_weight,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            # ... (add other relevant parameters)
        })

        for i in range(1, 1+args.iterations):
            content_images, _ = next(iter(content_dataloader))
            style_images, style_indices = next(iter(style_dataloader))

            style_codes = torch.zeros(args.batch_size, config.NUM_STYLE, 1)
            for b, s in enumerate(style_indices):
                style_codes[b, s] = 1

            content_images = content_images.to(device)
            style_images = style_images.to(device)
            style_codes = style_codes.to(device)

            output_images = model(content_images, style_codes)

            if isinstance(model, nn.DataParallel):
                content_features = loss_network(content_images.repeat(torch.cuda.device_count(), 1, 1, 1))
                style_features = loss_network(style_images.repeat(torch.cuda.device_count(), 1, 1, 1))
                output_features = loss_network(output_images.repeat(torch.cuda.device_count(), 1, 1, 1))
            else:
                content_features = loss_network(content_images)
                style_features = loss_network(style_images)
                output_features = loss_network(output_images)

            style_loss = calc_style_loss(output_features,
                                         style_features,
                                         config.STYLE_NODES)
            content_loss = calc_content_loss(output_features,
                                             content_features,
                                             config.CONTENT_NODES)
            tv_loss = calc_tv_loss(output_images)

            total_loss = content_loss \
                + style_loss * args.style_weight \
                + tv_loss * args.tv_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses['content'].append(content_loss.item())
            losses['style'].append(style_loss.item())
            losses['tv'].append(tv_loss.item())
            losses['total'].append(total_loss.item())

            # Log metrics
            mlflow.log_metrics({
                "content_loss": content_loss.item(),
                "style_loss": style_loss.item(),
                "tv_loss": tv_loss.item(),
                "total_loss": total_loss.item(),
            }, step=i)

            if i % 100 == 0:
                log = f"iter.: {i}"
                for k, v in losses.items():
                    # calculate a recent average value
                    avg = sum(v[-50:]) / 50
                    log += f", {k}: {avg:1.4f}"
                print(log)

        checkpoint_path = os.path.join('.', args.checkpoint_path)

        # Save the trained model
        torch.save({"state_dict": model.module.state_dict()}, checkpoint_path)

        # Log the trained model using mlflow.pytorch.log_model
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", registered_model_name="VGG16Model")

        # Plot losses
        plot_losses(losses, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training configurations
    parser.add_argument('--style_weight', type=float, default=config.STYLE_WEIGHT,
                        help='Weight for style loss')
    parser.add_argument('--tv_weight', type=float, default=config.TV_WEIGHT,
                        help='Weight for total variation loss')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--iterations', type=int, default=config.ITERATIONS,
                        help='Number of training iterations')

    # Data configurations
    parser.add_argument('--content_path', type=str, required=True,
                        help='Path to content images')
    parser.add_argument('--style_path', type=str, required=True,
                        help='Path to style images')
    parser.add_argument('--imsize', type=int, default=config.IMSIZE,
                        help='Input image size')
    parser.add_argument('--cropsize', type=int, default=config.CROPSIZE,
                        help='Crop size for input images')
    parser.add_argument('--cencrop', action='store_true',
                        help='Use center crop instead of random crop')

    # Other configurations
    parser.add_argument('--checkpoint_path', type=str, default='model.ckpt',
                        help='Path to save the trained model')

    args = parser.parse_args()

    train(args)