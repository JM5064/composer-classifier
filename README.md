# Composer-Classifier

Final project for McGill AI Society Intro to ML Bootcamp (Winter 2024)

Training data retreived from [Kaggle](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset).

## Project Overview
We used [librosa](https://librosa.org/) to turn the audio files into mel-spectrograms. We used a 6 layer Convoltuional Blocks that consists of a Conv2D with relu Activation, MaxPooling2D and a Dropout layer. It is followed by a flatten, dropout and dense layer with softmax activation. To evaluate the model, we used K-Fold Cross-Validation Technique to help increase generalizability of the model on unseen dataset. We also used F1_score to see the effectiveness of the model on classification and Balanced Accuracy Score  to see how effective it is with a skewed dataset.

## Run the program
To run the program, the packages need to be installed first. Download the dataset and put all the .wav files into /raw_data/all_audio, then run dataprocessing.py to convert the audio files into spectrograms. To run the model, run cnn.py.

## CNN Architecture
Our Architecture is based upon https://www.researchgate.net/publication/343320571_Efficient_Bone_Metastasis_Diagnosis_in_Bone_Scintigraphy_Using_a_Fast_Convolutional_Neural_Network_Architecture. We had to modify their architecture by changing the number of layers, implement K-Fold Cross-Validation as our dataset was much more skewed than in the research paper above and changing the hyperparameter values.