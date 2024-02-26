from pathlib import Path
from os import listdir
from os.path import join
import torch
import numpy as np
import cnn

torch.manual_seed(0)
np.random.seed(0)


train_path = Path(r"code/raw_data/spectrogram_train")
train_path = Path(r"/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/spectrogram_train/")
test_path = Path(r"code/raw_data/spectrogram_test")
test_path = Path(r"/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/spectrogram_test/")

train_files = listdir(train_path)
test_files = listdir(test_path)

train_loader = torch.utils.data.DataLoader(train_files, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_files, batch_size=4, shuffle=True)
# add validation set


# 320 train, 10 test
