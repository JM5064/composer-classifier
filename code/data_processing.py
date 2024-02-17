import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

path = Path(r"code/raw_data/audio/musicnet/train_data/1727.wav")
y, sr = librosa.load(path)
S = librosa.feature.melspectrogram(y=y, sr=sr)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
fig.savefig("test.png")
plt.show()
