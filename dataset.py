import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import sklearn.preprocessing as skp
import cv2

from torch.utils import data
from scipy.interpolate import splrep, splev
from biosppy.signals import tools as tools
import neurokit2 as nk
import scipy


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

def filter_ppg(signal, sampling_rate):
    
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4,#4, #
                                  frequency=[1, 8], #[0.5, 8]
                                  sampling_rate=sampling_rate)

    return filtered

def shuffle_filepath(paths):
    idx_list = [i for i in range(len(paths))]
    np.random.shuffle(idx_list)
    paths_shuffle = [paths[idx_list[i]] for i in range(len(paths))]
    
    return paths_shuffle

class PPG2ECG_Dataset(data.Dataset):
    def __init__(self, filepaths_ppg, filepaths_ecg, sampling_rate=128., min_max_norm=True, z_score_norm=True, interp='spline'):
#         self.filepaths = filepaths
        self.filepaths_ppg = filepaths_ppg#shuffle_filepath(filepaths_ppg)
        self.filepaths_ecg = self.filepaths_ppg#shuffle_filepath(filepaths_ecg)
        self.sampling_rate = sampling_rate
        self.min_max_norm = min_max_norm
        self.z_score_norm = z_score_norm
        
        self.interp = interp ## spline or cv2_linear
        
    def __len__(self):
        return len(self.filepaths_ppg)
    
    def prepare_data(self, index):
        # load data
#         data_dict = np.load(self.filepaths[index], allow_pickle=True).item()
        data_dict_ppg = np.load(self.filepaths_ppg[index], allow_pickle=True).item()
        data_dict_ecg = data_dict_ppg#np.load(self.filepaths_ecg[index], allow_pickle=True).item()
        
        original_ppg_fs = data_dict_ppg['PPG']['sig_fs']
        original_ppg_len = data_dict_ppg['PPG']['sig_len']
        original_ppg = data_dict_ppg['PPG']['sig']
        
        original_ecg_fs = data_dict_ecg['ECG']['sig_fs']
        original_ecg_len = data_dict_ecg['ECG']['sig_len']
 
        original_ecg = data_dict_ecg['ECG']['sig']
        rlabel = data_dict_ecg['ECG']['rlabel']


        # reshape
        ppg_seg = np.reshape(np.array(original_ppg), (1, -1))
        ecg_seg = np.reshape(np.array(original_ecg), (1, -1))
      #  rlabel_seg = np.reshape(np.array(rlabel), (1, -1))
      #  print(ppg_seg.shape, ecg_seg.shape)
        # z-score norm
        if self.z_score_norm == True:
            ppg_seg = (ppg_seg-ppg_seg.mean()) / (ppg_seg.std() + 1e-17)
            ecg_seg = (ecg_seg-ecg_seg.mean()) / (ecg_seg.std() + 1e-17)
        
        # self min-max normalize
        if self.min_max_norm == True:
            ppg_seg = skp.minmax_scale(ppg_seg, (-1, 1), axis=1)
            ecg_seg = skp.minmax_scale(ecg_seg, (-1, 1), axis=1)

        # ecg, ppg array to torch tensor
        ppg_seg = torch.from_numpy(ppg_seg).type(torch.FloatTensor)
        ecg_seg = torch.from_numpy(ecg_seg).type(torch.FloatTensor)
        rlabel = torch.from_numpy(rlabel).type(torch.IntTensor)
        data_dict = {'ppg': ppg_seg,
                     'ecg': ecg_seg,'rlabel': rlabel}
        
        return data_dict
    
    def __getitem__(self, index):
        return self.prepare_data(index)

class PPG2ECG_Dataset_Eval(data.Dataset):
    def __init__(self, filepaths, sampling_rate=128., min_max_norm=True, z_score_norm=True, interp='spline'):
#         self.filepaths = filepaths
        self.filepaths = filepaths
        self.sampling_rate = sampling_rate
        self.min_max_norm = min_max_norm
        self.z_score_norm = z_score_norm
        
        self.interp = interp ## spline or cv2_linear
        
     #   self.start_ratio = np.arange(0.05, 0.8, 0.05)
        
    def __len__(self):
        return len(self.filepaths)
    
    def prepare_data(self, index):
        # load data
        data_dict = np.load(self.filepaths[index], allow_pickle=True).item()
        
        original_ppg_fs = data_dict['PPG']['sig_fs']
        original_ppg_len = data_dict['PPG']['sig_len']
        original_ppg = data_dict['PPG']['sig']
        
        original_ecg_fs = data_dict['ECG']['sig_fs']
        original_ecg_len = data_dict['ECG']['sig_len']
        original_ecg = data_dict['ECG']['sig']
        
        
        rlabel = data_dict['ECG']['rlabel']

        # reshape
        ppg_seg = np.reshape(original_ppg, (1, -1))
        ecg_seg = np.reshape(original_ecg, (1, -1))
        
        # z-score norm
        if self.z_score_norm == True:
            ppg_seg = (ppg_seg-ppg_seg.mean()) / (ppg_seg.std() + 1e-17)
            ecg_seg = (ecg_seg-ecg_seg.mean()) / (ecg_seg.std() + 1e-17)
        
        # self min-max normalize
        if self.min_max_norm == True:
            ppg_seg = skp.minmax_scale(ppg_seg, (-1, 1), axis=1)
            ecg_seg = skp.minmax_scale(ecg_seg, (-1, 1), axis=1)
        
        # ecg, ppg array to torch tensor
        ppg_seg = torch.from_numpy(ppg_seg).type(torch.FloatTensor)
        ecg_seg = torch.from_numpy(ecg_seg).type(torch.FloatTensor)
        rlabel = torch.from_numpy(rlabel).type(torch.IntTensor)
        data_dict = {'ppg': ppg_seg,
                     'ecg': ecg_seg,'rlabel': rlabel}
        
        return data_dict
    
    def __getitem__(self, index):
        return self.prepare_data(index)