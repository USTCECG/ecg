# coding = utf-8
# 利用WFDB工具包将ECG数据可视化
from wfdb import *
from wfdb import processing
# 心电图.hea文件的存储位置
FilePath = "C://Software/IDE/Pycharm/MyProjectData/ECGData/100"
FileDirectory = "C://Software/IDE/Pycharm/MyProjectData/ECGData"
# sampleSingle, sampleField = rdsamp(FilePath, sampfrom=0, sampto=360)
# 打印出来的第一列是V5信号，第二类是MLII信号
# print(sampleSingle)
# plot_items(sampleSingle)
# 这里只能用plot_wfdb
# plot_wfdb(rdrecord(FilePath, sampfrom=1, sampto=100))
# plot_all_records需要.hea和.dat文件，一个都不能缺少
# plot_all_records(FileDirectory)
# anno = rdann(FilePath, extension="atr")

sig, fields = rdsamp(FilePath, channels=[0], sampfrom=0, sampto=510)
xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
xqrs.detect()
plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])







