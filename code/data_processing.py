import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join

train_path = Path(r"code/raw_data/audio/musicnet/train_data/")
train_path = Path(r"/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/audio/musicnet/train_data")
test_path = Path(r"code/raw_data/audio/musicnet/test_data/")
test_path = Path(r"/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/audio/musicnet/test_data")
train_files = listdir(train_path)
test_files = listdir(test_path)


def create_spectrograms(source_path, files):

    for filename in files:

        print(f"Converting {filename} now")

        # create melspectrogram
        y, sr = librosa.load(join(source_path, filename))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # create image
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        # turn off axis and remove padding
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # save into folder
        filename_parts = filename.split(".")
        output_name = "code/raw_data/spectrogram/" + filename_parts[0] + ".png"

        fig.savefig(output_name)
        plt.close()

    print("Done Converting!")


# Create the labels for composers
def create_labels(source_path, files, output_file_name):
    f = open("/Users/justinmao/Documents/GitHub/composer-classifier/raw_data/" + output_file_name, "a")

    for filename in files:
        file_number = int(filename[:4])

        composer = 0    # schubert

        if 1788 <= file_number <= 1893:
            composer = 1    # mozart
        elif 1916 <= file_number <= 1933:
            composer = 2    # dvorak
        elif 2075 <= file_number <= 2083:
            composer = 3    # cambini
        elif 2104 <= file_number <= 2106:
            composer = 4    # haydn
        elif 2112 <= file_number <= 2161:
            composer = 5    # brahms
        elif 2166 <= file_number <= 2169:
            composer = 6    # faure
        elif 2177 <= file_number <= 2180:
            composer = 7    # ravel
        elif 2186 <= file_number <= 2310:
            composer = 8    # bach
        elif 2313 <= file_number <= 2678:
            # there's one recording by bach with number 2659
            if file_number == 2659:
                composer = 8
            else:
                composer = 9    # beethoven

        f.write(str(composer) + "\n")
        print(filename)
    f.close()


# create_labels(train_path, train_files, "train_labels.txt")
# create_labels(test_path, test_files, "test_labels.txt")
# create_spectrograms(train_path, train_files)
# create_spectrograms(test_path, test_files)

