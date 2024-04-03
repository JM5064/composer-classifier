from pathlib import Path
from os import listdir
import torch
import torchmetrics as torchmetrics
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import dataset
import cnn

torch.manual_seed(0)
np.random.seed(0)


train_path = Path(r"raw_data/split_spectrogram_train")
test_path = Path(r"raw_data/split_spectrogram_test")
validation_path = Path(r"raw_data/split_spectrogram_validation")

train_files = listdir(train_path)
test_files = listdir(test_path)
validation_files = listdir(validation_path)

train_dataset = dataset.MusicDataset(train_path)
test_dataset = dataset.MusicDataset(test_path)
validation_dataset = dataset.MusicDataset(validation_path)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=True)
# 3,691 train, 196 validation, 46 test
num_composers = 10

# for data in train_loader:
#     print(data[1])
#     break


# set device to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# learning rate values to try
lr_values = {0.01, 0.1}
# number of iterations
num_epochs = 2


metrics = {}
models = {}

for lr in lr_values:
    model = cnn.CNN(num_channels=4).to(device)

    # initialize loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_composers).to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    metrics[lr] = {
        "losses": [],
        "accuracies": []
    }

    models[lr] = model

    for epoch in range(num_epochs):

        check_validation = 0

        for (X_train, y_train) in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_hat = model(X_train)
            loss = loss_function(y_hat, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            check_validation += 1

            if check_validation % 5 == 0:
                with torch.no_grad():
                    validation_loss = 0
                    validation_accuracy = 0

                    for X_val, y_val in validation_loader:
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)

                        y_hat = model(X_val)

                        validation_accuracy += accuracy(y_hat, y_val)
                        validation_loss += loss_function(y_hat, y_val)

                    validation_accuracy = validation_accuracy / len(validation_loader)
                    validation_loss = validation_loss / len(validation_loader)

                    metrics[lr]["losses"].append(validation_loss)
                    metrics[lr]["accuracies"].append(validation_accuracy)

                    print(f"LR = {lr} --- EPOCH = {epoch} --- ROUND = {check_validation}")
                    print(f"Validation loss = {validation_loss} --- Validation accuracy = {validation_accuracy}")



