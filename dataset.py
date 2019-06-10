import torch
import torchvision
from torch.utils.data import Dataset
import cv2


class AntispoofDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        image_info = self.paths[index]

        img = self.load_image(image_info['path'])
        if self.transform is not None:
            img = self.transform(img)

        return img, image_info['label']

    def __len__(self):
        return len(self.paths)

    def load_image(self, path):
        img = cv2.imread(path)
        return img