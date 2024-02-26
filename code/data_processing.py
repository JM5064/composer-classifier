import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from os import listdir
from os.path import join

source_path = Path(r"code/raw_data/audio/musicnet/")
files = listdir(source_path)

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
