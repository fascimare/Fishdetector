# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:02:47 2023

@author: P310512
"""

import opensoundscape
from opensoundscape import Audio, audio, Spectrogram
import math
import numpy
from matplotlib import pyplot as plt


path = 'G:/Shared drives/Wadden Sea Sound Library/Sound library/bubble growl/bubble growl_LAUW1_OFF_07_2023-05-18T231509.837Z.wav'
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
plt.plot(bpfrequencies,bpspectrum)
plt.ylabel('Fast Fourier Transform (V**2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.show()

from pathlib import Path

spectrogram_object = Spectrogram.from_audio(bandpassed,window_type='hann',window_samples=3500,overlap_fraction=0.9)
spec_bp = spectrogram_object.bandpass(50,1200)
spec_bp.plot(range=(-120,-70))