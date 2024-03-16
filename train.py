"""Training Script."""

import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

from configs import config
from models import StyleTransferNetwork, calc_content_loss, calc_style_loss, calc_tv_loss
from utils import ImageDataset, DataProcessor, imsave

def train(args):
    """Train Network."""
    device = torch.device('cuda')

    # data
    content_dataset = ImageDataset(dir_path=args.content_path)
    style_dataset = ImageDataset(dir_path=args.style_path)

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

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    losses = {'content': [], 'style': [], 'tv': [], 'total': []}
    print("Start training...")
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

        if i % 100 == 0:
            log = f"iter.: {i}"
            for k, v in losses.items():
                # calcuate a recent average value
                avg = sum(v[-50:]) / 50
                log += f", {k}: {avg:1.4f}"
            print(log)

    torch.save({"state_dict": model.state_dict()}, args.checkpoint_path)


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