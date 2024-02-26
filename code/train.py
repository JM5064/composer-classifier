from pathlib import Path
from os import listdir


source_path = Path(r"code/raw_data/audio/musicnet/train_data")
files = listdir(source_path)

# 320 train, 10 test
