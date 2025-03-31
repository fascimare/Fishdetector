# script to use best oss model for prediction
# and plot histograms


import matplotlib.pyplot as plt
import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import sklearn
import librosa
import torch
import wandb
import random
#from  apply_transfer_function import TransferFunction
#from convert_audio_to_bits import convert_audio_to_bits

model = opensoundscape.ml.cnn.load_model('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/second_try.model')
# for test data!!!! 
# in case test data is different sampling rate than training data (might just have to stick to one) yikes. What am I going to do about that? if loop? 
#model.preprocessor.pipeline.to_spec.params.window_samples = 1000 # 100 window samples
#model.preprocessor.pipeline.to_spec.params.overlap_samples = 900 # 90% overlap, for 2000 Fs this means 900 samples, and 0.05 sec bins
#model.preprocessor.pipeline.to_spec.params.fft_size = 2000 # FFT = Fs, 1 Hz bins

# load data 

test_clips = CINMS18B_clip
test_clips = pd.read_csv('C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/LAUW_eval_predictions_LAUW8.csv', index_col=[0,1,2])
test_clips = validate_clips
#test_clips = pd.read_csv('/home/michaela/CV4E/labeled_data/CINMS18B_one_hot_clips.csv', index_col=[0,1,2])
test_scores = model.predict(test_clips, num_workers=12,batch_size=128)
test_scores.columns = ['pred_bubblegrowl','pred_sfgrunt']
test_all = test_clips.join(test_scores)
test_evaluation = test_all.reset_index()

save_path = 'C:/Users/P310512/Documents/Groningen/Sound library/Fish sound clustering/TPWS library/LAUW_eval_predictions_4.csv'
test_evaluation.to_csv(save_path, index=False)

# bubble growl test
bg_eval_index = test_evaluation.index[test_evaluation['bubble growl']==1]
bg_eval = test_evaluation.loc[bg_eval_index]
bg_noise_index = test_evaluation.index[test_evaluation['bubble growl']==0]
bg_noise = test_evaluation.loc[bg_noise_index]

plt.hist(bg_noise['pred_bubblegrowl'],alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(bg_eval['pred_bubblegrowl'],alpha=0.5,edgecolor='black',color='orange',label='Bubble growl prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('Bubble growl prediction scores test')
plt.legend(loc='upper right') # this is progress. look at those high scoring missed detections
# blue is all of the examples in the D call column that did not actually contain a D call. 

# striped fish grunt test
fg_eval_index = test_evaluation.index[test_evaluation['grunt']==1]
fg_eval = test_evaluation.loc[fg_eval_index]
fg_noise_index = test_evaluation.index[test_evaluation['grunt']==0]
fg_noise = test_evaluation.loc[fg_noise_index]

plt.hist(fg_noise['pred_sfgrunt'],alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(fg_eval['pred_sfgrunt'],alpha=0.5,edgecolor='black',color='orange',label='Fish grunt prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('Fish grunt prediction scores test')
plt.legend(loc='upper right') 

