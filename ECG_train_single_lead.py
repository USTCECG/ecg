# -*- coding: utf-8 -*-
"""
Created on 2019

@author: xiajun 

# CPSC_train_single_lead.py: 训练针对指定导联的网络模型
单导联网络训练
心电信号有12个导联，这里是单导联的训练过程，后面我们会将12导联综合起来进行训练
"""

import os
import warnings
import numpy as np
from keras import optimizers
from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from ECG_model import Net
from ECG_config import Config
import ECG_utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

records_name = np.array(os.listdir(config.DATA_PATH))#记录文件的名字
records_label = np.load(config.REVISED_LABEL) - 1#记录的标签
class_num = len(np.unique(records_label))

# 划分训练，验证与测试集 -----------------------------------------------------------------------------------------------
#train表示训练集，test表示验证集，val表示测试集
train_val_records, test_records, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
del test_records, test_labels

train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

# 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
#调用了cspc-utils.py里面的oversample_balance函数
train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

# 取出训练集和测试集病人对应导联信号，并进行切片和z-score标准化 --------------------------------------------------------
print('Fetching data ...-----------------\n')
TARGET_LEAD = 1
# 对指定病人的单导联信号进行滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
# 第一个参数表示指定病人文件
train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
# 将类向量转变为二进制类矩阵
train_y = to_categorical(train_labels, num_classes=class_num)
# 对指定病人的单导联信号进行滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
# 第一个参数表示指定病人文件
val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
val_y = to_categorical(val_labels, num_classes=class_num)

# 最后生成的模型文件的名字
model_name = 'net_lead_' + str(TARGET_LEAD) + '.hdf5'

print('Scaling data ...-----------------\n')
for j in range(train_x.shape[0]):
    train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
for j in range(val_x.shape[0]):
    val_x[j, :, :] = scale(val_x[j, :, :], axis=0)

# 设定训练参数，搭建模型进行训练 （仅根据验证集调参，以及保存性能最好的模型）-------------------------------------------
batch_size = 64
epochs = 100 # 训练轮次
momentum = 0.9
keep_prob = 0.5

inputs = Input(shape=(config.SEG_LENGTH, config.SEG_NUM))
net = Net()# CPSC_model文件里面的函数
outputs, _ = net.nnet(inputs, keep_prob, num_classes=class_num)# 全连接层前自动提取的特征
model = Model(inputs=inputs, outputs=outputs)
#随机梯度下降法优化
opt = optimizers.SGD(lr=config.lr_schedule(0), momentum=momentum)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH+model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
lr_scheduler = LearningRateScheduler(config.lr_schedule)
callback_lists = [checkpoint, lr_scheduler]
model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(val_x, val_y), callbacks=callback_lists)

del train_x, train_y
#加载模型
model = load_model(config.MODEL_PATH + model_name)
#进行预测，在验证集上进行预测，val_x表示预测集
pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(val_y, axis=1)
del val_x, val_y

# 评估模型在验证集上的性能 ---------------------------------------------------------------------------------------------
Conf_Mat_val = confusion_matrix(true_v, pred_v)
print('Result-----------------------------\n')
print(Conf_Mat_val)
F1s_val = []
for j in range(class_num):
    f1t = 2 * Conf_Mat_val[j][j] / (np.sum(Conf_Mat_val[j, :]) + np.sum(Conf_Mat_val[:, j]))
    print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
    F1s_val.append(f1t)

print('F1-mean: ' + str(np.mean(F1s_val)))
