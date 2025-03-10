# 基于mamba特征提取与attention特征融合建模的青少年早期抑郁症检测算法

## 模型结构图"
![image](https://github.com/user-attachments/assets/41324575-aaee-44c5-a6dc-379433e95896)

## 实验结果:
### 第一次由于算力（显存的限制）,我们在序列长度上做了截断，截断分别为1024、2048和4096，第一次实验采取了三个模态的信息进行融合，max_length截断设置为1024。如下图所示，详情见源码：script/dataset
![image](https://github.com/user-attachments/assets/651e7604-e820-4041-8f89-a770c961a16f)

###第二次，加大截断，将max_length设置为2048 eeg和ecg双模态融合,并且做了4的切片
![image](https://github.com/user-attachments/assets/c59619a0-5be5-43bb-8cca-66d6e928423f)
 
第三次，于gpu得到更新，第三次加大截断，将max_length设置为4096 eeg和ecg双模态融合,同样做4的切片。
第四次，截断不变仍然为(4096)，此次试验切片为8.
具体的实验次数远大于上面所描述的，由于算力限制，故先只对以上四次实验结果进行分析。
