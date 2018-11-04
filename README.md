
USTC-SSE-2018-2019-Engineering Practice-ECG Recognition Group

#### MIT-BIH数据库简介
来源：
[http://www.physionet.org/physiobank/database/mitdb/](http://www.physionet.org/physiobank/database/mitdb/) <br>
MIT-BIH数据库包含了48条数据，它们都是两个声道的流动性ECG数据。在1975到1979年从47人身上获取，其中23条随机的选自一个集合（编号100到124）：4000条时长为24个小时的住院病人（6成）或者非住院病人（4成）的数据。另外的25条（编号200到234）选自同一个集合，它们不是那么常见，但是在临床上的意义重大。
每个声道每秒360个采样点，11比特的分辨率，电压范围10毫伏，多名心脏病专家对数据做了注解
。采样人群为25名年龄为32到89的男性，以及22名年龄23到89的女性，其中201和202号数据是同一个男性病例。
<br>
MIT-BIH数据库的使用请参考：[https://blog.csdn.net/chenyusiyuan/article/details/2027887](https://blog.csdn.net/chenyusiyuan/article/details/2027887)
<br>
<br>
以下来源：[https://www.physionet.org/physiobank/database/html/mitdbdir/intro.htm](https://www.physionet.org/physiobank/database/html/mitdbdir/intro.htm)<br>
大部分信号上面的信道是修改过的MLII（一种导联方式）信号，下面的信道通常是V1（偶尔是V2、V5，有一个是V4）信号，QRS波群通常在上面的信道比较突出，正常的心电节拍在下层信道是很难差察觉的。

#### 如何将读取的MIT-BIH的数据可视化
使用官方提供的python类库(WFDB)解析下载的ECG数据即可。<br>
WFDB：[https://github.com/MIT-LCP/wfdb-python](https://github.com/MIT-LCP/wfdb-python)<br>
API：[https://wfdb.readthedocs.io/en/latest/index.html](https://wfdb.readthedocs.io/en/latest/index.html)<br>
需要先pin install wfdb，然后编写一段python代码读取你下载的数据文件(.hea结尾的）。

