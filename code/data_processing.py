import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join
import os
import soundfile as sf

train_path = Path(r"code/raw_data/audio/musicnet/train_data/")
test_path = Path(r"code/raw_data/audio/musicnet/test_data/")
train_files = listdir(train_path)
test_files = listdir(test_path)


# split each wav file into segments and export them to destination_directory
def split_wav_files(files, source_path, destination_path, split_duration):
    for file in files:
        audio_data, sample_rate = sf.read(str(source_path) + "/" + file)

        num_splits = int((len(audio_data) / sample_rate) // split_duration)
        print(file[0:4])

        for i in range(num_splits):
            start = sample_rate * split_duration * i
            end = sample_rate * split_duration * (i+1)
            split_section = audio_data[start:end]

            output_file = os.path.join(destination_path, f"{file[0:4]}_{i}.wav")
            sf.write(output_file, split_section, sample_rate)


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

# create_labels() NOT NEEDED
# 0 - Schubert
# 1 - Mozart
# 2 - Dvorak
# 3 - Cambini
# 4 - Haydn
# 5 - Brahms
# 6 - Faure
# 7 - Ravel
# 8 - Bach
# 9 - Beethoven

# Create the labels for composers
def create_labels(files, output_file_name):
    f = open("code/raw_data/" + output_file_name, "a")

    for filename in files:
        file_number = int(filename[:4])

        composer = 0  # schubert

        if 1788 <= file_number <= 1893:
            composer = 1  # mozart
        elif 1916 <= file_number <= 1933:
            composer = 2  # dvorak
        elif 2075 <= file_number <= 2083:
            composer = 3  # cambini
        elif 2104 <= file_number <= 2106:
            composer = 4  # haydn
        elif 2112 <= file_number <= 2161:
            composer = 5  # brahms
        elif 2166 <= file_number <= 2169:
            composer = 6  # faure
        elif 2177 <= file_number <= 2180 or 2802 <= file_number <= 2808:  # no ravel in test?
            composer = 7  # ravel
        elif 2186 <= file_number <= 2310:
            composer = 8  # bach
        elif 2313 <= file_number <= 2678:
            # there's one recording by bach with number 2659
            if file_number == 2659:
                composer = 8
            else:
                composer = 9  # beethoven

        f.write(str(composer) + "\n")
        print(filename)
    f.close()


split_wav_files(test_files, test_path, "composer-classifier/raw_data/audio/musicnet/split_test_data", 30)
split_wav_files(test_files, test_path, "composer-classifier/raw_data/audio/musicnet/split_test_data", 30)
# create_labels(train_files, "train_labels.txt")
# create_labels(test_files, "test_labels.txt")
# create_spectrograms(train_path, train_files)
# create_spectrograms(test_path, test_files)
