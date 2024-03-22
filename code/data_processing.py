import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join
import os
import soundfile as sf

train_path = Path(r"code/raw_data/audio/musicnet/split_train_data/")
test_path = Path(r"code/raw_data/audio/musicnet/split_test_data/")
train_files = listdir(train_path)
test_files = listdir(test_path)


# split each wav file into segments and export them to destination_directory
def split_wav_files(files, source_path, destination_path, split_duration):
    for file in files:
        # read wav file
        audio_data, sample_rate = sf.read(str(source_path) + "/" + file)

        num_splits = int((len(audio_data) / sample_rate) // split_duration)
        print(file[0:4])

        # create splits
        for i in range(num_splits):
            start = sample_rate * split_duration * i
            end = sample_rate * split_duration * (i + 1)
            split_section = audio_data[start:end]

            # write file to destination directory
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
        output_name = "code/raw_data/split_spectrogram_train/" + filename_parts[0] + ".png"

        fig.savefig(output_name)
        plt.close()

    print("Done Converting!")


split_wav_files(test_files, test_path, "composer-classifier/raw_data/audio/musicnet/split_test_data", 30)
split_wav_files(test_files, test_path, "composer-classifier/raw_data/audio/musicnet/split_test_data", 30)
create_spectrograms(train_path, train_files)
create_spectrograms(test_path, test_files)
