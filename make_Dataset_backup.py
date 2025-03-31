# script to make opensoundscape datasets for training, validation, and test!
# will be modified later once I have more data

import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import sklearn
import librosa
import torch
import random

# read in the files that you want 

# smoosh all of DCPP data together for train and validate datasets
DCPP1 = pd.read_csv('G:/My Drive/Wadden fish sounds/Data analysis/Final logs/modified_annotations/LAUW_05_allsites_newformat_inputmod2_modification.csv')
DCPP2 = pd.read_csv('G:/My Drive/Wadden fish sounds/Data analysis/Final logs/modified_annotations/LAUW_06_allsites_inputmod2_modification.csv')
DCPP3 = pd.read_csv('G:/My Drive/Wadden fish sounds/Data analysis/Final logs/modified_annotations/LAUW_07_allsites_inputmod2_modification.csv')

DCPP_all = pd.concat([DCPP1,DCPP2,DCPP3],ignore_index=True)
DCPP_all_box = opensoundscape.BoxedAnnotations(DCPP_all)
DCPP_all_box = opensoundscape.BoxedAnnotations(DCPP_all)
DCPP_all_box.audio_files =  DCPP_all['audio_file'].unique()

#creating one-hot-clips for all data, joining them together, and then random splitting. 
DCPP_clips = DCPP_all_box.one_hot_clip_labels(clip_duration=5,clip_overlap=0,min_label_overlap=2)
#DCPP_all_A = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['A NE Pacific'])
#DCPP_all_B = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['B NE Pacific'])

# overlap was different for different calls, now I have to join all of the rows together based on their columns
#new = DCPP_all_D.join(DCPP_all_A)
#DCPP_clips = new.join(DCPP_all_B)
DCPP_clips.to_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/LAUW_one_hot_clips2_5s.csv', index=True)

train_clips, validate_clips = sklearn.model_selection.train_test_split(DCPP_clips, train_size=0.7, random_state=0) # use this function to randomly subset them and spit out two new dataframes
#path_to_remove = '/mnt/ssd-cluster/michaela/data/xwavs/DCCP01A_fall/DCPP01A_d01_121115_054102.d100.x.wav'
#train_clips = train_clips.reset_index()
#train_clips_new = train_clips[train_clips['file'] != path_to_remove] # this will need to be modified for column indices
balanced_train_clips = opensoundscape.data_selection.resample(train_clips,n_samples_per_class=1000,random_state=0) # upsample (repeat samples) so that all classes have 1000 samples
#balanced_train_clips_standard = balanced_train_clips.reset_index()


# must
#train_clips_new = train_clips_new.reset_index(drop=True)
#train_clips_new
#filtered_indices = train_clips_new.index[(train_clips_new['D'] == 0) & (train_clips_new['A NE Pacific'] == 0) & (train_clips_new['B NE Pacific'] == 0)]# indices of negatives

#random_sample_indices = random.sample(filtered_indices.tolist(), 1500)

#combined_indices = list(random_sample_indices) 

# and now reapply my filtered indices to the dataframe 
#train_clips_filtered = train_clips[combined_indices]
#train_clips_noise = train_clips_new.iloc[random_sample_indices]

#train_clips_final = pd.concat([train_clips_noise, balanced_train_clips_standard]).reset_index(drop=True)

#train_clips_final.sum()

#validate_clips = validate_clips.reset_index()
#validate_clips_new = validate_clips[validate_clips['file'] != path_to_remove]
#validate_clips_final = validate_clips_new.reset_index(drop=True)
#Make sure that sample size of validation labels doesn't get too small
validate_bgonly = validate_clips[validate_clips['bubble growl']==1]
validate_sfonly = validate_clips[validate_clips['grunt']==1]
validate_noise = validate_clips[validate_clips['bubble growl']==0]
#validate_noise = validate_clips[validate_clips['grunt']==0]
validate_noise2 = validate_noise[validate_noise['grunt']==0]
validate_clips_sub = pd.concat([validate_bgonly,validate_sfonly,validate_noise2.iloc[0:10000]])
#validate_clips_sub = pd.concat([validate_sfonly,validate_noise.iloc[0:10000]])
validate_random_samples = validate_clips_sub.sample(n=2000, random_state=1)

# now save each of these as a csv for training! 
validate_random_samples.to_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/validate_clips2_15s.csv', index=True)
#validate_clips.to_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/validate_clips_gr_15s.csv', index=True)
balanced_train_clips.to_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/train_clips2_15s.csv', index=True)

# now do the same with test data
CINMS18B = pd.read_csv('G:/My Drive/Wadden fish sounds/Data analysis/Final logs/modified_annotations/LAUW_08_allsites_inputmod2_modification.csv')
CINMS18B_box = opensoundscape.BoxedAnnotations(CINMS18B)
CINMS18B_box.audio_files =  CINMS18B['audio_file'].unique()

CINMS18B_clip = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=2)
#CINMS18B_clip_gr = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=2,class_subset=['grunt'])

#new_clip = CINMS18B_clip_bg.join(CINMS18B_clip_gr)
CINMS18B_clip.to_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/LAUW_test_one_hot_clips2_15s.csv', index=False)
