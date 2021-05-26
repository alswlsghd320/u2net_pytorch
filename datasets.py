from torch.utils.data import Dataset
from transforms import train_transforms, test_transforms
from PIL import Image
import os

def filename_without_ext(file_path):
    basename = os.path.basename(file_path)
    filename = os.path.splitext(basename)[0]
    return filename

class RmbgDataset_from_txt(Dataset):
    def __init__(self, img_txt_path, mask_txt_path, resize_shape=(1280, 1080), crop_shape=(1024,1024), is_train=True):

        self.img_list = self.make_list_from_txt(img_txt_path)
        self.mask_list = self.make_list_from_txt(mask_txt_path)
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.is_train = is_train

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')

        if self.is_train:
            img, mask = train_transforms(img, mask, self.resize_shape, self.crop_shape)
        else:
            img, mask = test_transforms(img, mask, self.resize_shape)

        return [img, mask]

    def make_list_from_txt(self, file_path):
        f = open(file_path, 'r')
        lines = []
        while True:
            line = f.readline()
            if not line: break
            lines.append(line.rstrip())
        f.close()

        return lines

class RmbgDataset_from_path(Dataset):
    def __init__(self, img_path, mask_path, resize_shape=(1280, 1080), crop_shape=(1024,1024), is_train=True):

        self.img_path = img_path
        self.mask_path = mask_path
        self.file_list = os.listdir(self.img_path)
        self.img_list = [os.path.join(self.img_path, i) for i in self.file_list]
        self.mask_list = [os.path.join(mask_path, filename_without_ext(i)+'.png') for i in self.file_list]
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.is_train = is_train

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')

        if self.is_train:
            img, mask = train_transforms(img, mask, self.resize_shape, self.crop_shape)
        else:
            img, mask = test_transforms(img, mask, self.resize_shape)

        return [img, mask]

def RmbgDataset(cfg, img_path, mask_path, resize_shape=(1280, 1080), crop_shape=(1024,1024), is_train=True):
    dataset_type = cfg['dataset_type']
    if dataset_type == 'folder':
        return RmbgDataset_from_path(img_path, mask_path, resize_shape, crop_shape, is_train)
    elif dataset_type == 'txt':
        return RmbgDataset_from_txt(img_path, mask_path, resize_shape, crop_shape, is_train)
    else:
        ValueError('dataset_type must be "folder" or "txt"')
