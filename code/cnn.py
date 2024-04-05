import numpy as np
from pathlib import Path
from keras import layers
from keras import models
import keras
from keras.utils import to_categorical
import tensorflow as tf
from PIL import Image
from numpy import asarray
import os
import csv
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_score

# ----------- CONVERTING IMAGES INTO ARRAYS -----------

train_path = Path(r"raw_data/audio/train/")
train_files = os.listdir(train_path)
test_path = Path(r"raw_data/audio/test/")
test_files = os.listdir(test_path)

train_images_list = []
test_images_list = []
train_uncoded_labels = []
test_uncoded_labels = []
labels = {}

with open("raw_data/musicnet_metadata.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        labels[row[0]] = row[1]
        first_line = False

for spec in train_files:
    audio_full_name = os.path.basename(spec)
    audio_basename = audio_full_name.split("_")[0]
    audio_label = labels[audio_basename]
    train_uncoded_labels.append(audio_label)
    img = Image.open("raw_data/audio/train/" + audio_full_name)
    img = img.convert("RGB")
    new_img = img.resize((256, 256))
    numpydata = asarray(new_img)
    train_images_list.append(numpydata)
    
train_images = np.array(train_images_list)
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_uncoded_labels)

#print(train_images.shape)
#print(train_labels.shape)

for spec in test_files:
    audio_full_name = os.path.basename(spec)
    audio_basename = audio_full_name.split("_")[0]
    audio_label = int(labels[audio_basename])
    test_uncoded_labels.append(audio_label)
    img = Image.open("raw_data/audio/test/" + audio_full_name)
    img = img.convert("RGB")
    new_img = img.resize((256, 256))
    numpydata = asarray(new_img)
    test_images_list.append(numpydata)

test_images = np.array(test_images_list)
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_uncoded_labels)

#print(test_images.shape)
#print(test_labels.shape)

# ----------- CREATING CNN MODEL -----------

batch_size = 32
epochs = 50

model = models.Sequential()

#Layer 1
model.add(layers.Conv2D(16, (3,3), activation = 'relu', input_shape= (256,256,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

#Layer 2
model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

#Layer 3
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

#Layer 4
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

#Layer 5
model.add(layers.Conv2D(256, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

#Layer 6
model.add(layers.Conv2D(512, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))


model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128))
model.add(layers.Dense(10, activation='softmax'))

#model.summary()

# ----------- RUNNING THE MODEL -----------

callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

opt = keras.optimizers.SGD(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])

kf = KFold(n_splits=5, shuffle=True, random_state=1) 

scores = []
for train_index, test_index in kf.split(train_images):
    print("Train:", train_index, "Validation:",test_index)
    X_train, X_test = train_images[train_index], train_images[test_index] 
    y_train, y_test = train_labels[train_index], train_labels[test_index]

    history = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[callback],
            validation_split=0.20,
            verbose=1)
    
    y_pred = model.predict(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_test_bool = np.argmax(y_test, axis=1)
    fold_score = balanced_accuracy_score(y_test_bool, y_pred_bool)
    
    scores.append(fold_score)
    
mean_accuracy = sum(scores)/len(scores)

# ----------- PLOTTING TRAINING DATA -----------

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("model_accuracy.png")
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("model_loss.png")

# ----------- CALCULATING MATRICIES -----------

target_labels = ["Schubert","Mozart","Dvorak","Cambini","Haydn","Brahms","Faure","Ravel","Bach","Beethoven"]
y_pred = model.predict(test_images)
y_pred_bool = np.argmax(y_pred, axis=1) 
bas = balanced_accuracy_score(test_uncoded_labels, y_pred_bool)

print(classification_report(test_uncoded_labels, y_pred_bool, target_names=target_labels, zero_division=0))
print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", mean_accuracy)
print(f"Balanced Accuracy Score: {bas}\n")
