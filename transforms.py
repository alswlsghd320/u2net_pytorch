import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

def train_transforms(img, mask, resize_shape=(1280, 1080), crop_shape=(1024,1024)):
    img = transforms.Resize(resize_shape, interpolation = Image.BILINEAR)(img)
    mask = transforms.Resize(resize_shape, interpolation = Image.NEAREST)(mask)

    i, j, h, w = transforms.RandomCrop.get_params(img, crop_shape)

    img = TF.crop(img, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    img = TF.to_tensor(img)
    mask = TF.to_tensor(mask)

    img = transforms.Normalize(0.5, 0.5)(img)

    return img, mask

def test_transforms(img, mask, resize_shape=(1024,1024)):
    img = transforms.Resize(resize_shape, interpolation = Image.BILINEAR)(img)
    mask = transforms.Resize(resize_shape, interpolation = Image.NEAREST)(mask)
    img = TF.to_tensor(img)
    img = transforms.Normalize(0.5, 0.5)(img)
    mask = TF.to_tensor(mask)

    return img, mask
