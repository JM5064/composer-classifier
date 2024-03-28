import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join
from os import rename
import soundfile as sf
import random

train_path = Path(r"raw_data/audio/musicnet/train_data/")
train_files = listdir(train_path)

split_train_path = Path(r"raw_data/audio/musicnet/split_train_data/")
split_train_files = listdir(split_train_path)

test_path = Path(r"raw_data/audio/musicnet/test_data/")
test_files = listdir(test_path)

split_test_path = Path(r"raw_data/audio/musicnet/split_test_data/")
split_test_files = listdir(split_test_path)


random.seed(0)


# split each wav file into segments and export them to destination_directory
def split_wav_files(files, source_path, destination_path, split_duration):
    for file in files:
        # read wav file
        # audio_data - the samples
        # sample_rate - number of samples per second
        audio_data, sample_rate = sf.read(str(source_path) + "/" + file)

        num_splits = int((len(audio_data) / sample_rate) // split_duration)

        remaining_frames = len(audio_data) - split_duration * num_splits * sample_rate
        begin = random.randint(0, remaining_frames // sample_rate)

        # +1 for unexplainable reasons
        start = begin * sample_rate + 1
        print(file[0:4], begin, start)

        # create splits
        for i in range(num_splits):
            end = start + sample_rate * split_duration - 1
            split_section = audio_data[start:end]

            # write file to destination directory
            output_file = join(destination_path, f"{file[0:4]}_{i}.wav")
            sf.write(output_file, split_section, sample_rate)

            start += sample_rate * split_duration


def create_spectrograms(source_path, files, destination_path):
    for filename in files:
        if filename.endswith(".wav"):
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
            output_name = destination_path + "/" + filename_parts[0] + ".png"

            fig.savefig(output_name)
            plt.close()

    print("Done Converting!")


def get_validation_spectrograms():
    train_spectrograms_path = Path(r'raw_data/split_spectrogram_train/')
    train_spectrograms_files = listdir(train_spectrograms_path)

    validation_spectrogram_path = Path(r'raw_data/split_spectrogram_validation/')

    frequency = 0
    for filename in train_spectrograms_files:
        # take every 20th file from train data and move into validation set
        # maybe not the best way to make validation set
        if frequency % 20 == 0:
            # move file to validation folder
            destination_file = join(validation_spectrogram_path, filename)
            rename(str(train_spectrograms_path) + "/" + filename, destination_file)

        frequency += 1


# UNCOMMENT THESE TO CREATE DATA

# # Run these two to split the .wav files into 30 second segments
# split_wav_files(train_files, train_path, "raw_data/audio/musicnet/split_train_data", 30)
# split_wav_files(test_files, test_path, "raw_data/audio/musicnet/split_test_data", 30)
#
# # Run these two to convert the split .wav files into spectrograms
# create_spectrograms(split_train_path, split_train_files, "raw_data/split_spectrogram_train")
# create_spectrograms(split_test_path, split_test_files, "raw_data/split_spectrogram_test")
#
# # Run to create validation set from train data
# get_validation_spectrograms()
