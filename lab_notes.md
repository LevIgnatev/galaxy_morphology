## 2025-08-31
Added an improved, more detailed description of the project in the README (still to be improved). Planning on downloading the data tomorrow.

## 2025-09-01
Downloaded the galaxy zoo 2 dataset (~3 GB) and the Sloan Digital Sky Survey (DR18 and 17) (a total of ~1 GB)

## 2025-09-05
Decided to use the galaxy zoo 2 dataset. With the help of AI (results that this task is incredibly difficult due to the data format) over the last couple of days managed to save the mapped csv with data. 
A snippet was saved to data/labels. The manifest with labels consists of 92 columns representing features of every image, as well as the scores in each category (spiral, smooth, merger, etc.)
The last column corresponds to the final derived label, with a threshold of certainty of 0.6 (if not met - the galaxy is classified as ambiguous).

## 2025-09-12
Over the course of the last week built a baseline classifier, as well as a data pipeline for preprocessing (train_classifier.py and data_loader.py respectively).
The model used is CNN (convolutional neural network) with a ResNet50 pretrained base (the starting configuration of layers is rather arbitrary, optimal parameters will be introduced later).

## 2025-09-15
Continued working on the first notebook (data exploration). Added an extensive description and visuals of the properties of the dataset.

## 2025-09-22
Finished the first notebook a couple of days ago. Also split the entire dataset in two parts: the one used for training and the one for the final evaluation.

## 2025-10-06
Second notebook on the preprocessing (02_preproc.ipynb) added + 03_baselines notebook started. Improved baseline model to avoid overfitting. Started working on the baseline captioner.

## 2025-10-07
Baseline classifier improved (fixed even, I would say). 3rd notebook extended greatly. Generated captions using varied, pseudo-random templates and saved them to captions_full.csv. Turned out to be much, much harder than expected.
Started working on explaining and interpreting the classifier's predictions using heat maps (just a stub for now).

Difficulty: my nvidia rtx 4060 was enough for experiments, but running full models on the entire dataset takes dozens of hours. Might look into using a remote GPU server.

## 2025-10-10
app.py - an interactive streamlit demo added. Minor improvements.

## 2025-11-07
Haven't written here in a long time, but the last few weeks were spent on the captioner module.
Got it to a more or less working condition. Added it to the demo.
Cleaned the repo, biggest change: removal of hardcoded Windows paths in favour of 'Path's from pathlib
(mainly for reproducibility).
Minor improvements to the notebooks.

## 2025-11-12
Trained the classifier and the captioner on the full datasets using cloud GPUs from runpod. Probably the hardest task to date. At least I learned some CLI commands. Saved the weights to checkpoints. Fixed the app and the scripts.

## 2025-11-24
Final repo cleanup.