# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:57:08 2023

@author: P310512
"""

import opensoundscape
from pathlib import Path 
from opensoundscape.preprocess.actions import BaseAction
import pandas as pd
import math

class Sensitivity(BaseAction):
    """Apply hydrophone sensitivity to spectrogram object
    """
    
    def __init__(self, decibel_limits=None):
        super(Sensitivity,self).__init__()
        self.params['decibel_limits']=decibel_limits 

        sens_path = 'G:/My Drive/Sound library/Sound types Wadden Sea/Hydrophone calibration.xlsx'
        self.Sens = pd.read_excel(sens_path)
        
    def go(self,sample,**kwargs):
        path = Path(sample.source)
        deployment_name = path.name.split('.')[0]
        deploymentfloat = float(deployment_name)

        #cal = self.Sens['Sensitivity high gain'].loc['Hydrophone'==deployment_name]
        cal = self.Sens.loc[self.Sens['Hydrophone'] == deploymentfloat, 'Sensitivity high gain'].values[0]

        sample.data = apply_sensitivity_value(sample.data, cal, self.params['decibel_limits'])    
        
def apply_sensitivity_value(spec,cal,decibel_limits=None):
    """
    apply sensitivity to opensoundscape.Spectrogram object
    
    helper function to apply calibration from sensitivity to Spectrogram
    sensitivity is dB offset value
  

    Args:
        spec: a Specrogram object
        cal: calibration/sensitivity value in dB (should be a positive number)
        decibel_limits: default None will use original spectrogram's .decibel_units attribute;
            optionally specify a new decibel_limits range for the returned Spectrogram
    """
    if decibel_limits is None:
        decibel_limits = spec.decibel_limits
    new_spec_values = spec.spectrogram+cal
    # add the offset values to each row of the spectrogram
   # new_spec_values = spec.spectrogram*math.pow(10,cal/10)
      
    #create a new spectrogram object with the new values
    return opensoundscape.Spectrogram(new_spec_values,times=spec.times,frequencies=spec.frequencies)