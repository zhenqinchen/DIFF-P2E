import time
import sys

import torch

import numpy as np
import neurokit2 as nk
from biosppy.signals import ecg as ecg_func
from biosppy.signals import tools as tools
import neurokit2.ppg as ppg_func


def get_Rpeaks_ECG(filtered, sampling_rate):
    
    # segment
    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg_func.correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    # extract templates
    templates, rpeaks = ecg_func.extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    rr_intervals = np.diff(rpeaks)

    return rpeaks, rr_intervals


def heartbeats_ecg(filtered, sampling_rate):
    
    rpeaks, rr_intervals = get_Rpeaks_ECG(filtered, sampling_rate)

    if rr_intervals.size != 0:
    # compute heart rate
        hr_idx, hr = tools.get_heart_rate(beats=rpeaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)

        if len(hr)==0:
            hr_idx, hr = [-1], [-1]
            
    else:
        hr_idx, hr = [-1], [-1]

    
    return hr_idx, hr

def heartbeats_ppg(filtered, sampling_rate):
    
    peaks, peaks_intervals = get_peaks_PPG(filtered, sampling_rate)

    if peaks_intervals.size != 0:
        # compute heart rate
        hr_idx, hr = tools.get_heart_rate(beats=peaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)

        if len(hr)==0:
            hr_idx, hr = [-1], [-1]
        
    else:
        hr_idx, hr = [-1], [-1]
        
    return hr_idx, hr

def get_peaks_PPG(filtered, sampling_rate=128):
    
    # segment
    peaks = ppg_func.ppg_findpeaks(filtered, sampling_rate)['PPG_Peaks']
    peak_intervals = np.diff(peaks)
   
    return peaks, peak_intervals


def ecg_bpm_array(ecg_signal, sampling_rate=128, window=4, filter=False):

    final_bpm = []
    for k in ecg_signal:
        if filter == True:
            k = nk.ecg_clean(k, sampling_rate=sampling_rate, method="pantompkins1985")
        hr_idx, hr = heartbeats_ecg(k, sampling_rate)
        # print(hr)
        bpm = np.mean(hr)
        final_bpm.append(bpm)    
    return np.array(final_bpm)   
def ppg_bpm_array(ppg_signal, sampling_rate=128, window=4):
    
    final_bpm = []
    # count=0
    for k in ppg_signal:

        try:
            hr_idx, hr = heartbeats_ppg(k, sampling_rate)
            # print(count)
            bpm = np.mean(hr)
            final_bpm.append(bpm)    
            # count=count+1
        except:
            final_bpm.append(-1.0)

    return np.array(final_bpm) 

def MAE_hr(ppg, real_ecg, fake_ecg, sampling_freq=128, window_size=4):

     ######################## HR estimation from Fake ECG ######################

    real_ecg_bpm = ecg_bpm_array(real_ecg, sampling_freq, window_size)
    fake_ecg_bpm = ecg_bpm_array(fake_ecg, sampling_freq, window_size, filter=False) ## check for -1 values
    
    ppg_bpm = ppg_bpm_array(ppg, sampling_rate=sampling_freq, window=window_size)
    ## correction
    fbpm = fake_ecg_bpm[np.where(fake_ecg_bpm != -1)]
    rbpm = real_ecg_bpm[np.where(fake_ecg_bpm != -1)]
    pbpm = ppg_bpm[np.where(ppg_bpm != -1)]
    rrbpm = real_ecg_bpm[np.where(ppg_bpm != -1)]
    mae_hr_ecg = np.mean(np.absolute(rbpm - fbpm))
    mae_hr_ppg = np.mean(np.absolute(rrbpm - pbpm))
    return mae_hr_ecg, mae_hr_ppg


