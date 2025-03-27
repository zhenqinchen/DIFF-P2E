from yacs.config import CfgNode as CN
import numpy as np

config = CN()
###数据位置###
config.model_save_path = 'E:/resourse/2024/ppg2ecg/model'
config.partition_path = "D:/research/data/2024/PPG2ECG/partition/01_BIDMC_Partition.npy"

###数据库###
config.RESULT='E:/resourse/2024/ppg'
config.ARTICLE_RESULT='E:/resourse/2024/ppg/result'


config.db = ''



config.smooth = False 
config.seg_len = int(4*128) ##输入模型的长度
config.fs = 128 
config.smooth_label = False 
config.filter = [7.5, 75]
config.z_score_norm = True
config.sig_time_len = 4
config.interp_method = 'spline'#
config.min_max_norm = True


##model
config.num_worker = 1
config.train = True 

config.batch_size = 192 
config.max_epoch = 80 
config.loss_name = 'focal'
config.ckpt = 'ckpt' 
config.lr =0.0002#0.0001


config.loss_weight = 0.5
config.sample_k = 1

config.b1 = 0.5
config.b2 = 0.999


##evaluate
config.seed = 1


 

def cfg():
    return config