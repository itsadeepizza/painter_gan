from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import itertools
import glob, os

class ImageDataset(Dataset):
    """ Dataloader for Monet paintings AND photos"""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # List of all jpg images contained in the directory
        self.images = [file for file in os.listdir(img_dir) if file.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        image = image.to(dtype=torch.float) / 255
        return image

    def get_img(self, idx):
        """Only for plotting"""
        img = self[idx].squeeze().swapaxes(0, 2).swapaxes(0, 1)
        return img
