
# script to load triton log annotations and modify them to fit opensoundscapes format
# needs filepath, call label, annotation start and end time
from datetime import datetime
import os
import glob
import opensoundscape
#from AudioStreamDescriptor import XWAVhdr
from opensoundscape import Audio, Spectrogram
import random
import pandas as pd
import numpy as np

directory_path = "G:/My Drive/Wadden fish sounds/Data analysis/Final logs/"
all_files = glob.glob(os.path.join(directory_path,'*inputmod2.xlsx'))

# function to extract xwav start time and save it
#def extract_xwav_start(path):
 #   xwav_hdr = XWAVhdr(path)
  #  xwav_start_time = xwav_hdr.dtimeStart
   # return xwav_start_time

# function to get annotation start and end time in seconds since start of xwav
# also replaces old file path with new one
# removes fin whale calls
# uses extract_xwav_start to get get file start time for each row

#new_path_LAUW1OFF="F:/WaddenSea fish sounds/Lauwersoog recordings/May1/LAUW1_OFF_06/"

# calculate start and end time of annotation in seconds since start of xwav
def calculate_annotation_seconds(df):
    df['audio_file'] = [in_file.replace("\\","/") for in_file in df['new_inputfile']] # list comprehension for swapping out file path
    #df['file_datetime'] = df['audio_file'].apply(extract_xwav_start) # use apply function to apply extract_xwav_datetime to all rows
    df['nameparts'] = df['new_inputfile'].str.split('.')
    #df['fileparts'] = df['Input file'].str.split('\\')
    #def combine_strings(row):
    #    return f"{row['fileparts'][0]}_{row['Location']}"
    #df['new_column'] = df.apply(combine_strings, axis=1)
    df['filetimestr'] = df['nameparts'].apply(lambda x: x[1] if len(x) > 1 else None)
    df['file_datetime'] = pd.to_datetime(df['filetimestr'],format='%y%m%d%H%M%S',errors='coerce')
    df['start_time'] = (df['StartTime'] - df['file_datetime']).dt.total_seconds() # convert start time difference to total seconds
    df['end_time'] = (df['EndTime'] - df['file_datetime']).dt.total_seconds() # convert end time difference to total seconds
    #bp_indices = df.index[df['Species Code'] == 'Bp'].tolist() # indices of fin whale calls
    #df.drop(bp_indices, inplace=True)  #remove fin whale calls
    #noise_indices = df.index[df['Species Code'] == 'Na'].tolist() # indices of noise
    #df.drop(noise_indices, inplace=True)  #remove noise annotations 
    df['annotation']= df['Comments']
    #df['high_f'] = df['Parameter 1']
    #df['low_f'] = df['Parameter 2']
    #df['Input file'] = [in_file.replace("\\","/").replace("E:/SocalLFDevelopmentData/",new_path) for in_file in df['Input file']] # list comprehension for swapping out file path
    df = df.loc[:, ['audio_file','annotation','start_time','end_time']] # subset all rows by certain column name
    return df

# make a subfolder for saving modified logs 
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# loop through all annotation files and save them in subfolder "modified_annotations"
#new_path="/mnt/ssd-cluster/michaela/data/xwavs/"

for file in all_files:
    file2 = file.replace("\\~$","/")
    data = pd.read_excel(file2)
    subset_df = calculate_annotation_seconds(data)
    maincalls_df = subset_df[(subset_df['annotation'] == "grunt")] #maincalls_df = subset_df[(subset_df['annotation'] == "bubble growl") | (subset_df['annotation'] == "grunt")]
    filename = os.path.basename(file)
    new_filename = filename.replace('.xlsx', '_gr_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    maincalls_df.to_csv(save_path, index=False)
    

