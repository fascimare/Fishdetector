# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:02:47 2023

@author: P310512
"""

import opensoundscape
from opensoundscape import Audio, audio, Spectrogram
import sys
import math
import numpy
from matplotlib import pyplot as plt
import sys
sys.path.append('C:/Users/P310512/Documents/GitHub/opensoundscape')


path = 'F:/Sound library/bubble growl/bubble growl_LAUW1_OFF_07_2023-05-18T231509.837Z.wav'
audio_object = Audio.from_file(path)
audio_object.metadata



audio_segment = Audio.from_file(file,offset = 30.0, duration = 1.0)
audio_segment.duration

#Calculate fft
fft_spectrum, frequencies = audio_object.trim(0,5).spectrum()
cal = 177
calspectrum = fft_spectrum*math.pow(10,cal/10)
dbspectrum = 10*numpy.log10(calspectrum)

#bandpass
bandpassed = audio_object.bandpass(low_f=100,high_f=2000,order=12)
bpspectrum, bpfrequencies = bandpassed.spectrum()

plt.rcParams['figure.figsize']=[15,5]
%config InlineBackend.figure_format = 'retina'

# plot
plt.plot(frequencies,dbspectrum)
plt.ylabel('Fast Fourier Transform (V**2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.show()

from pathlib import Path

spectrogram_object = Spectrogram.from_audio(bandpassed,window_type='hann',window_samples=3500,overlap_fraction=0.9)
spec_bp = spectrogram_object.bandpass(50,1200)
spec_bp.plot(range=(-180,-70))

from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.utils import collate_audio_samples_to_tensors
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5]
pre = SpectrogramPreprocessor(sample_duration=5.0)
labels = balanced_train_clips

dataset = AudioFileDataset(labels,pre)
dataset.bypass_augmentations = True
tensors = [dataset[i].data for i in range(9)]
sample_labels = [list(dataset[i].labels[dataset[i].labels>0].index) for i in range(9)]

_ = show_tensor_grid(tensors,3,labels=sample_labels)

#Seeing what the model will see with the preprocessing
preprocessor = SpectrogramPreprocessor(sample_duration=5.0)

# moodify model preprocessing for making spectrograms the way I want them
preprocessor.pipeline.to_spec.params.window_type = 'hamming' # using hamming window (Triton default)
preprocessor.pipeline.to_spec.params.window_samples = 3500 # 1600 window samples
preprocessor.pipeline.to_spec.params.overlap_fraction = 0.9 # 90% overlap, for 3200 Fs this means 1400 samples, and 0.05 sec bins
preprocessor.pipeline.to_spec.params.fft_size = 3500 # FFT = Fs, 1 Hz bins
#preprocessor.pipeline.to_spec.params.decibel_limits = (50,100) # oss preprocessing sets dB limits. These get reset when tf is applied
preprocessor.pipeline.to_spec.params.scaling = 'spectrum'
preprocessor.pipeline.to_tensor.params.range = (50,100)
preprocessor.pipeline.bandpass.params.min_f = 50
preprocessor.pipeline.bandpass.params.max_f = 1200
preprocessor.pipeline.add_noise.bypass=True
#preprocessor.pipeline.frequency_mask.set(max_width = 0.03, max_masks=10)
#preprocessor.pipeline.time_mask.set(max_width = 0.1, max_masks=10)
preprocessor.pipeline.add_noise.set(std=0.2)
preprocessor.pipeline.random_affine.bypass=True
#preprocessor.height = 224
#preprocessor.width = 448
#preprocessor.channels = 3
preprocessor.out_shape = [224,448,3]

import sys
sys.path.append('C:/Users/P310512/Documents/GitHub/Fishdetector')
from apply_sensitivity_value import Sensitivity

preprocessor.insert_action(
    action_index='apply_cal', #give it a name
    action= Sensitivity(decibel_limits=(50,100)), #the action object
    after_key='to_spec') #where to put it (can also use before_key=...)

dataset2 = AudioFileDataset(labels,preprocessor)

show_tensor(dataset2[1].data)
plt.show()

labels = train_clips
file = labels.iloc[618].name
sample = preprocessor.forward((file[0],file[1]),trace=True)
file = train_clips['file'].iloc[618]
sttime = train_clips['start_time'].iloc[618]
sample = preprocessor.forward((file,sttime), trace=True)
sample.trace

sample.trace["to_spec"].plot(range=(-120,-70))
sample.trace["bandpass"].plot()
show_tensor(sample.trace["time_mask"])
show_tensor(sample.trace["rescale"])

dataset2.bypass_augmentations = True
tensors = [dataset2[i].data for i in range(9)]
sample_labels = [list(dataset2[i].labels[dataset2[i].labels>0].index) for i in range(9)]

_ = show_tensor_grid(tensors,3,labels=sample_labels)
