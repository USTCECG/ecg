# -*- coding: utf-8 -*-
"""
Created on 2019

@author: ZhaoJunWen

# CPSC_hybrid.py: 使用xgboost混合深度学习网络和人工特征，得到最后结果

"""

import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ECG_config import Config
import ECG_utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = Config()
config.MODEL_PATH = 'G:/EngineerPractice/ECG20190221/ECG_SIGNAL/Net_models/'
config.MAN_FEATURE_PATH = 'G:/EngineerPractice/ECG20190221/ECG_SIGNAL/Man_features/'


records_name = np.array(os.listdir(config.DATA_PATH))
#print('zjw++++++++++++++++++',records_name.shape)
records_label = np.load(config.REVISED_LABEL) - 1
#print('zjw++++++++++++++++++',records_label.shape)
class_num = len(np.unique(records_label))


# 下面这两部分数据集的划分，在这个类里根本没用到
train_val_records, _, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

_, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
_, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

# 载入之前保存的网络输出概率以及人工特征 -------------------------------------------------------------------------------
pred_nnet_r = np.load(config.MODEL_PATH + 'pred_nnet_r.npy')
pred_nnet_v = np.load(config.MODEL_PATH + 'pred_nnet_v.npy')
pred_nnet_t = np.load(config.MODEL_PATH + 'pred_nnet_t.npy')

man_features_r = np.load(config.MAN_FEATURE_PATH + 'man_features_r.npy')
man_features_v = np.load(config.MAN_FEATURE_PATH + 'man_features_v.npy')
man_features_t = np.load(config.MAN_FEATURE_PATH + 'man_features_t.npy')


pred_r = np.concatenate((pred_nnet_r, man_features_r), axis=1)
pred_v = np.concatenate((pred_nnet_v, man_features_v), axis=1)
pred_t = np.concatenate((pred_nnet_t, man_features_t), axis=1)

lb_r = train_labels
lb_v = val_labels
lb_t = test_labels

# print(pred_t)
# print(pred_t.shape)
# print(lb_t)
# print(lb_t.shape)

# 训练xgboost （仅根据验证集表现调参）----------------------------------------------------------------------------------
dtrain = xgb.DMatrix(pred_r, label=lb_r)
dval = xgb.DMatrix(pred_v, label=lb_v)

# 这里是把测试集的数据量控制为1  *******************
patient=int(input('这里有1376个病人的ecg数据，请输入你要判断的病人的序号：'))
print('正在判断，请稍等......')
pred_t=pred_t[patient-1:patient][:]
lb_t=lb_t[patient-1:patient]
dtest = xgb.DMatrix(pred_t, label=lb_t)




# 疾病类型与代号的映射关系
jibing_daihao_mapping={0:'正常',1:'心房颤动',2:'I度房室阻滞',3:'左束支阻滞',4:'心脏为非健康状态，具体疾病需进一步判定',5: '心脏为非健康状态，具体疾病需进一步判定',6:'心脏为非健康状态，具体疾病需进一步判定',7:'心脏为非健康状态，具体疾病需进一步判定',8:'心脏为非健康状态，具体疾病需进一步判定'}


param = [('max_depth', 10), ('objective', 'multi:softmax'),
         ('eval_metric', 'merror'), ('subsample', 0.5),
         ('eta', 0.01), ('num_class', 9), ('min_child_weight', 1.0),
         ('gamma', 0.2), ('colsample_bytree', 0.7),
         ('lambda', 20), ('max_delta_step', 7)
         ]

watchlist = [(dtrain, 'train'), (dval, 'test')]
num_round = 25
bst = xgb.train(param, dtrain, num_round, watchlist)

# 评估在过采样验证集，原始验证集，以及测试集上的性能 -------------------------------------------------------------------
# pred = bst.predict(dval)
# Conf_Matv = confusion_matrix(lb_v, pred)
#
# print('\nResult for oversampling val_set:--------------------\n')
# F1s_val = []
# for j in range(Conf_Matv.shape[0]):
#     f1vt = 2*Conf_Matv[j][j]/(np.sum(Conf_Matv[j, :])+np.sum(Conf_Matv[:, j]))
#     print('| F1-'+config.CLASS_NAME[j]+':'+str(f1vt)+' |')
#     F1s_val.append(f1vt)
# print('\nF1-mean: ' + str(np.mean(F1s_val)))
#
# val_records_raw, val_records_ind = np.unique(val_records, return_index=True)
# lb_v_raw = lb_v[val_records_ind]
# pred_raw = pred[val_records_ind]
#
# Conf_Matv_raw = confusion_matrix(lb_v_raw,pred_raw)
# print('\nResult for raw val_set:--------------------\n')
# F1s_val_raw = []
# for j in range(Conf_Matv_raw.shape[0]):
#     f1vt = 2*Conf_Matv_raw[j][j]/(np.sum(Conf_Matv_raw[j, :])+np.sum(Conf_Matv_raw[:, j]))
#     print('| F1-'+config.CLASS_NAME[j]+':'+str(f1vt)+' |')
#     F1s_val_raw.append(f1vt)
# print('\nF1-mean: ' + str(np.mean(F1s_val_raw)))

pred = bst.predict(dtest)

print('\n 测试结果为:--------------------\n')
print('病人的  实际  心脏疾病类型为：',jibing_daihao_mapping[int(lb_t[0])])
print('\n')
print('病人的  预测  心脏疾病类型为：',jibing_daihao_mapping[int(pred[0])])
if int(lb_t[0])==int(pred[0]):
    print('\n预测成功！！！')
else:
    print('\n预测失败！！！')


# Conf_Matt = confusion_matrix(lb_t, pred)
#
# #print('\n 测试结果为:--------------------\n')
# F1s_test = []
# for j in range(Conf_Matt.shape[0]):
#     f1tt = 2*Conf_Matt[j][j]/(np.sum(Conf_Matt[j, :])+np.sum(Conf_Matt[:, j]))
#     print('| F1-'+config.CLASS_NAME[j]+':'+str(f1tt)+' |')
#     F1s_test.append(f1tt)
# print('\nF1-mean: ' + str(np.mean(F1s_test)))
