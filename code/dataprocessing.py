import csv
import os
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join
from os import rename
import soundfile as sf
import random
import shutil

all_audio_path = Path(r"raw_data/audio/all_audio/")
all_audio_files = listdir(all_audio_path)

split_audio_path = Path(r"raw_data/audio/split_audio/")
split_audio_files = listdir(split_audio_path)

spectrograms_path = Path(r"raw_data/audio/spectrograms/")
spectrogram_files = listdir(spectrograms_path)

train_path = Path(r"raw_data/audio/train/")
test_path = Path(r"raw_data/audio/test/")

        
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
            
            
def sep_spectrograms():
    counter = 0
    dir_test = 'raw_data/audio/test/'
    dir_train = 'raw_data/audio/train/'
    print("---------------- START --------------------")
    for spec in spectrogram_files:
        audio_full_name = os.path.basename(spec)
        org_file_path = "raw_data/audio/spectrograms/" + audio_full_name
                
        if (counter == 4):
            # test data
            shutil.move(org_file_path, dir_test)
            counter = 0
        else:
            # training data
            shutil.move(org_file_path, dir_train)
            counter += 1
            
    print("---------------- DONE --------------------")
        
split_wav_files(all_audio_files, all_audio_path, "raw_data/audio/split_audio", 30)
create_spectrograms(split_audio_path, split_audio_files, "raw_data/audio/spectrograms")
sep_spectrograms()