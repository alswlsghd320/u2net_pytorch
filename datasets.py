from torch.utils.data import Dataset, DataLoader
from transforms import train_transforms, test_transforms
from PIL import Image

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