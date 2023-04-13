from torch.utils.data import Dataset
from PIL import Image
import os


class Carvana(Dataset):

    def __init__(self, path, transforms, train=True):
        self.path = path
        self.transforms = transforms
        self.images = sorted([x for x in os.listdir(self.path + 'train')])
        self.masks = sorted([x for x in os.listdir(self.path + 'train_masks')])
        if train:
            self.images = self.images[:int(len(self.images) * 0.7)]
            self.masks = self.masks[:int(len(self.masks) * 0.7)]
        else:
            self.images = self.images[int(len(self.images) * 0.7):]
            self.masks = self.masks[int(len(self.masks) * 0.7):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.path + 'train/' + self.images[idx]).convert('RGB')
        mask = Image.open(self.path + 'train_masks/' + self.masks[idx]).convert('L')
        return self.transforms(image), self.transforms(mask)