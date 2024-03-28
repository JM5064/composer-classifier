from pathlib import Path
from os import listdir
from os.path import join
import torch
import torch.nn as nn
import numpy as np
import cnn

torch.manual_seed(0)
np.random.seed(0)


train_path = Path(r"code/raw_data/split_spectrogram_train")
test_path = Path(r"code/raw_data/split_spectrogram_test")
validation_path = Path(r"code/raw_data/split_spectrogram_validation")

train_files = listdir(train_path)
test_files = listdir(test_path)
validation_files = listdir(validation_path)

train_loader = torch.utils.data.DataLoader(train_files, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_files, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_path, batch_size=4, shuffle=True)
# 3,691 train, 196 validation, 46 test


# learning rate values to try
lr_values = {0.01, 0.1}
# number of iterations
num_epochs = 2

# define loss function
loss = nn.CrossEntropyLoss()



