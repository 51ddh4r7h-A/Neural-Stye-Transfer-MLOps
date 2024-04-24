"""Image Utility Code."""

import torchvision
import torchvision.transforms as T
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

normalize = T.Normalize(mean=MEAN, std=STD)
denormalize = T.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                          std=[1/std for std in STD])


def get_transforms(imsize=None, cropsize=None, cencrop=False):
    """Get the transforms."""
    transformer = []
    if imsize:
        transformer.append(T.Resize(imsize))
    if cropsize:
        if cencrop:
            transformer.append(T.CenterCrop(cropsize))
        else:
            transformer.append(T.RandomCrop(cropsize))

    transformer.append(T.ToTensor())
    transformer.append(normalize)
    return T.Compose(transformer)


def imload(path, imsize=None, cropsize=None, cencrop=False):
    """Load a image."""
    transformer = get_transforms(imsize=imsize,
                                 cropsize=cropsize,
                                 cencrop=cencrop)
    image = Image.open(path).convert("RGB")
    return transformer(image).unsqueeze(0)


def imsave(image, save_path):
    """Save a image."""
    image = denormalize(torchvision.utils.make_grid(image)).clamp_(0.0, 1.0)
    torchvision.utils.save_image(image, save_path)
    return None