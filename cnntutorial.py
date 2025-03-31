# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:22:48 2023

@author: P310512
"""

#CNN tutorial

from opensoundscape import CNN
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess
from glob import glob
import sklearn
from pathlib import Path 

#set up plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5]
%config InlineBackend.figure_format = 'retina'

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# Set the current directory to where the dataset is downloaded
dataset_path = Path("./Documents/Github/Fishdetector/annotated_data/")

# Make a list of all of the selection table files
selection_files = glob(f"{dataset_path}/annotation_Files/*/*.txt")

# Create a list of audio files, one corresponding to each Raven file
# (Audio files have the same names as selection files with a different extension)
selection_files = [f.replace("\\","/") for f in selection_files]
audio_files = [f.replace('annotation_Files','wav_Files').replace('.Table.1.selections.txt','.wav') for f in selection_files]

from opensoundscape.annotations import BoxedAnnotations
# Create a dataframe of annotations
annotations = BoxedAnnotations.from_raven_files(
    selection_files,
    audio_files)


# Parameters to use for label creation
clip_duration = 3
clip_overlap = 0
min_label_overlap = 0.25
species_of_interest = ["NOCA","EATO","SCTA","BAWW","BCCH","AMCR","NOFL"]

# Create dataframe of one-hot labels
clip_labels = annotations.one_hot_clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
    class_subset = species_of_interest # You can comment this line out if you want to include all species.
)

clip_labels.head()

# Select all files from Recording_4 as a test set
mask = clip_labels.reset_index()['file'].apply(lambda x: 'Recording_4' in x).values
test_set = clip_labels[mask]

# All other files will be used as a training set
train_and_val_set = clip_labels.drop(test_set.index)

# Save .csv tables of the training and validation sets to keep a record of them
train_and_val_set.to_csv("./Documents/Github/Fishdetector/annotated_data/train_and_val_set.csv")
test_set.to_csv("./Documents/Github/Fishdetector/annotated_data/test_set.csv")

# Split our training data into training and validation sets
train_df, valid_df = sklearn.model_selection.train_test_split(train_and_val_set, test_size=0.1, random_state=0)

from opensoundscape.data_selection import resample

# upsample (repeat samples) so that all classes have 800 samples
balanced_train_df = resample(train_df,n_samples_per_class=800,random_state=0)

# Create a CNN object designed to recognize 3-second samples
from opensoundscape import CNN

# Use resnet34 architecture
architecture = 'resnet34'

# Can use this code to get your classes, if needed
class_list = list(train_df.columns)

model = CNN(architecture = architecture, classes = class_list, sample_duration = clip_duration)

print(f'model.device is: {model.device}')

import wandb

wandb_session = wandb.init(entity = 'mareco', project = 'OpenSoundscape tutorials', name = 'Train CNN')

checkpoint_folder = Path("./Documents/Github/Fishdetector/annotated_data/model_training_checkpoints")
checkpoint_folder.mkdir(exist_ok=True)

model.train(
    balanced_train_df, valid_df,
    epochs = 2, batch_size = 64, log_interval = 100,
    num_workers = 4,  save_interval =10,
    save_path = checkpoint_folder)