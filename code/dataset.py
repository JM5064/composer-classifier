import torch
from torch.utils.data import Dataset
import pandas as pd
from os import listdir


class MusicDataset(Dataset):
    def __init__(self, path):
        # the folder with the image files of the dataset
        self.path = str(path)
        self.files = listdir(self.path)

        # load csv file for labels
        self.df = pd.read_csv("/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/musicnet_metadata_short.csv")
        self.labels = self.df[['composer']]

        self.dataset = torch.tensor(self.labels.to_numpy().reshape(-1).long())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image_path = self.path + "/" + str(self.files[item])
        image_num = self.files[item][:4]

        label = self.df.loc[image_num, 'composer']

        return image_path, label

