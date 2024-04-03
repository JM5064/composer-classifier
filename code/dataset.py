import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from os import listdir
from PIL import Image


class MusicDataset(Dataset):
    def __init__(self, path):
        # the folder with the image files of the dataset
        self.path = str(path)
        self.files = listdir(self.path)

        # load csv file for labels
        self.df = pd.read_csv("raw_data/musicnet_metadata.csv")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.37,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image_path = self.path + "/" + str(self.files[item])

        image = Image.open(image_path)
        image = self.transform(image)

        image_num = int(self.files[item][:4])
        label = torch.tensor(self.df.loc[self.df['id'] == image_num, 'composer_id'].values[0])

        return image, label

