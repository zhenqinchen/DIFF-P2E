import time,os
import numpy as np
import torch
import random
from torch.optim import lr_scheduler

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import dtw
from biosppy.signals import tools as tools
import neurokit2 as nk
import HR_util
import pandas as pd
from openpyxl import load_workbook
from scipy.signal import find_peaks, butter, filtfilt
import file_util as fu

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    
def speed_up():

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    



    
def PPC(y, y_pred):
    p = stats.pearsonr(y, y_pred)[0]
    return np.abs(p)
    
def MAE(y, y_pred):
    return mean_absolute_error(y, y_pred)
    

    



def PRD(y, y_pred):
    numerator = ((y-y_pred)**2).sum()
    denominator = (y**2).sum()
    
    return np.sqrt((numerator / (denominator + 1e-10))*100)

def DTW(y, y_pred):
    return dtw.dtw(y, y_pred, keep_internals=True).distance

def normalize_signal(signal):
    """ 归一化信号 """
    return (signal - np.mean(signal)) / np.std(signal)

def cross_correlation(signal1, signal2):
    """ 计算两个信号的互相关系数 """
    # 归一化信号
    signal1_normalized = normalize_signal(signal1)
    signal2_normalized = normalize_signal(signal2)
    
    # 计算互相关
    correlation = np.correlate(signal1_normalized, signal2_normalized, 'full')
    # 归一化相关系数
    n = len(signal1)
    m = len(signal2)
    denom = n * np.std(signal1) * np.std(signal2)
    normalized_correlation = correlation / denom
    
    max_corr = np.max(normalized_correlation)
    lag = np.argmax(normalized_correlation) - n + 1
    
    return np.abs(max_corr)


def fid_features_to_statistics(features):
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        'mu': mu,
        'sigma': sigma,
    }




def filter_ppg(signal, sampling_rate):
    
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4, #3
                                  frequency=[1, 8], #[0.5, 8]
                                  sampling_rate=sampling_rate)

    return filtered


def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    
    return filtered


def filter_signal(ppg, ecg , fs, ecg_fs = None):
#     ppg = np.nan_to_num(ppg)
#     ecg = np.nan_to_num(ecg)
    ppg = nk.ppg_clean(ppg, sampling_rate=fs)
    if ecg_fs is None:
        ecg_fs = fs
    ecg = nk.ecg_clean(ecg, sampling_rate=ecg_fs, method="pantompkins1985")
    ppg = np.array(ppg)
    ecg = np.array(ecg)
    return ppg,ecg


def filter_signal_2(ppg, ecg , fs, ecg_fs = None):
    ppg = np.nan_to_num(ppg)
    ecg = np.nan_to_num(ecg)
    if ecg_fs is None:
        ecg_fs = fs
    ppg = filter_ppg(ppg, fs)
    ecg = filter_ecg(ecg, ecg_fs)
    ppg = np.array(ppg)
    ecg = np.array(ecg)
    return ppg,ecg


def calc_hr(peaks, fs=250):
    rr_inter = []
    tmp = 0
    for x in peaks:
        if tmp == 0:
            tmp = x
            continue
        rr_inter.append(60 / (abs(x - tmp) / fs))
        tmp = x
    return np.mean(rr_inter)

def calc_ppg_hr(ppg_sig, fs, peak_method='elgendi'):
    """
    method : "elgendi", "bishop"
    """
    ppg_peak = nk.ppg_findpeaks(ppg_sig, sampling_rate=fs, method=peak_method)['PPG_Peaks']
    hr = calc_hr(ppg_peak, fs)
    
    return hr

def calc_ecg_hr(ecg_sig, fs, peak_method='nabian2018'):
    """
    method : "nabian2018", "neurokit"
    """
    ecg_peak = nk.ecg_peaks(ecg_sig, sampling_rate=fs, method="nabian2018")[1]['ECG_R_Peaks']
    hr = calc_hr(ecg_peak, fs)
    
    return hr
def hr_MAE(y, y_pred):
    return mean_absolute_error(y, y_pred)

import matplotlib.pyplot as plt
from util import MAE, RMSE, DTW,PPC
def evaluate( noisy,refs, preds, rlabel, c, align = False):
    
    MAE_temp= []

    RMSE_temp = []

    DTW_temp = []
    PPC_temp = []
 #   CC_temp = []
  
    
    HR_GT = []
    HR_ppg = []
    HR_pred = []
    is_show = False
    for i in range(refs.shape[0]):

        ref,syn, rpeak,origin = refs[i], preds[i],rlabel[i], noisy[i]
        ref = np.nan_to_num(ref)
        syn = np.nan_to_num(syn)

        ref = nk.ecg_clean(ref, sampling_rate=c.fs, method="pantompkins1985")
        syn = nk.ecg_clean(syn, sampling_rate=c.fs, method="pantompkins1985")
        ref = np.array(ref)
        syn = np.array(syn)
        gt_hr = calc_ecg_hr(ref, fs=c.fs,peak_method = 'neurokit')
        ppg_hr = calc_ppg_hr(origin, fs=c.fs, peak_method='bishop')
        pred_hr = calc_ecg_hr(syn, fs=c.fs,peak_method = 'neurokit')

        if align:
            ref,syn = align_ecg_signals(ref, syn, max_delay_ms=100, fs=c.fs)        
        


        MAE_temp.append(MAE(ref, syn))

        RMSE_temp.append(RMSE(ref, syn))

        DTW_temp.append(DTW(ref, syn))
        PPC_temp.append(PPC(ref,syn))
        
        if np.isnan(gt_hr) or np.isnan(ppg_hr) or np.isnan(pred_hr):
            continue
        HR_GT.append(gt_hr)
        HR_ppg.append(ppg_hr)
        HR_pred.append(pred_hr)
    
    HR_GT = np.array(HR_GT)
    HR_ppg = np.array(HR_ppg)
    HR_pred = np.array(HR_pred)

    

    model_perf_dict = {}
    model_perf_dict['model_name'] = c.model_file
    model_perf_dict['PPC'] = np.array(PPC_temp).mean() 
    model_perf_dict['RMSE'] = np.array(RMSE_temp).mean()
    model_perf_dict['DTW'] = np.array(DTW_temp).mean() 
    model_perf_dict['PPC_std'] = np.array(PPC_temp).std() 
    model_perf_dict['RMSE_std'] = np.array(RMSE_temp).std()
    model_perf_dict['DTW_std'] = np.array(DTW_temp).std() 
    model_perf_dict['K'] = c.sample_k
    
    model_perf_dict['align'] = str(align)


    print(model_perf_dict)


        
    

    
from scipy.signal import correlate   
def align_ecg_signals(test_ecg, predict_ecg, max_delay_ms=100, fs=1000):
    """
    对齐两个 ECG 信号，计算延迟不超过指定的最大延迟（毫秒）。

    参数：
    test_ecg : numpy.ndarray
        真实 ECG 信号。
    predict_ecg : numpy.ndarray
        预测 ECG 信号。
    max_delay_ms : int
        最大允许延迟（毫秒）。
    fs : int
        采样频率（Hz）。

    返回：
    tuple : (test_ecg_aligned, predict_ecg_aligned)
        对齐后的真实和预测 ECG 信号。
    """
    max_delay_samples = int((max_delay_ms / 1000) * fs)

    # 计算互相关，只考虑最大延迟范围内的样本
    c = correlate(test_ecg, predict_ecg, mode='full')
    lags = np.arange(-len(predict_ecg) + 1, len(test_ecg))

    # 仅选择在最大延迟范围内的延迟值
    valid_indices = np.where(np.abs(lags) <= max_delay_samples)[0]
    c_valid = c[valid_indices]
    lags_valid = lags[valid_indices]

    # 找到最大相关系数的索引
    lag_i = np.argmax(c_valid)
    lag = lags_valid[lag_i]

    # 根据延迟调整信号
    if lag < 0:
        test_ecg_aligned = test_ecg[:len(test_ecg) + lag]
        predict_ecg_aligned = predict_ecg[-lag:]
    else:
        predict_ecg_aligned = predict_ecg[:len(predict_ecg) - lag]
        test_ecg_aligned = test_ecg[lag:]

    return test_ecg_aligned, predict_ecg_aligned

from scipy.interpolate import splrep, splev
from biosppy.signals import tools as tools
import scipy

def interp_spline(ecg, step=1, k=3):
    x_new = np.arange(0, ecg.shape[0], ecg.shape[0]/step)
    interp_spline_method = splrep(np.arange(0, ecg.shape[0], 1), ecg, k=k)
    return splev(x_new, interp_spline_method)
def resample(signal, origin_fs, fs, interp='spline'):
   # signal = signal.copy()
    if interp == 'spline':
        sig_seconds = len(signal) // origin_fs
        signal = interp_spline(signal, step=fs*sig_seconds, k=5)
    elif interp == 'cv2_linear':
        sig_seconds = len(signal) // origin_fs
        signal = cv2.resize(signal, (1, fs*sig_seconds), interpolation=cv2.INTER_LINEAR).flatten()
    elif interp == 'scipy':
        scale_fs = origin_fs/fs
        signal = scipy.signal.resample(signal, int(len(signal) / scale_fs), axis=0)
    return signal



# R峰检测函数
def detect_r_peaks(ecg_signal, fs):
    # 滤波参数
    lowcut = 5.0  # 低截止频率 (Hz)
    highcut = 15.0  # 高截止频率 (Hz)
    ecg_filtered =ecg_signal# bandpass_filter(ecg_signal, lowcut, highcut, fs, order=2)
    
    # 差分
    ecg_diff = np.diff(ecg_filtered)
    
    # 平方
    ecg_squared = ecg_diff ** 2
    
    # 移动窗口积分
    window_size = int(0.150 * fs)  # 150毫秒的窗口
    ecg_integrated = np.convolve(ecg_squared, np.ones(window_size)/window_size, mode='same')
    
    # 检测R峰
    threshold = np.mean(ecg_integrated)
    peaks, _ = find_peaks(ecg_integrated, height=threshold, distance=0.3*fs)
    
    return peaks#, ecg_integrated