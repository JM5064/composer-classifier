import torch
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
        self.labels = self.df[['composer']]

        self.dataset = torch.tensor(self.labels.to_numpy().reshape(-1).long())

        self.transform = torch.transforms.Compose([
            torch.transforms.ToTensor(),
            torch.transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image_path = self.path + "/" + str(self.files[item])

        image = Image.open(image_path)
        image = self.transform(image)

        image_num = self.files[item][:4]
        label = self.df.loc[image_num, 'composer']

        return image, label
