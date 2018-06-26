---

## o98k啊黎 ##
### 欧文杰 ###
### https://zhuanlan.zhihu.com/p/38499275  ###
---
## 运行介绍 ##

环境要求
 - 180G以上内存
 - 500G以上磁盘存储空间
 - 10G以上的cuda显卡
 - python 2.7
 - pytorch 0.2.0
 
运行说明
``` 
./run.sh
```

参数修改
```
vim src/args.py
```
 
---

## 文件夹说明 ##
|-- data (存放数据文件)

	|-- chusai (保存初赛的原始数据)

	|-- fusai  (保存复赛的原始数据)

	(data中下列的文件夹都是脚本自动生成)

	|-- bin\_files (numpy保存格式)

	|-- feature2idx (存放字典数据)

	|-- infos (存放辅助用数据)

|-- src  (所有代码文件,具体代码文件在下一部分介绍)

	|-- logs (保存log输出文件)

|-- models (保存pytorch模型)

|-- result (保存结果文件)


---

## 代码文件说明 ##
	(1) args.py  参数以及一些常量的文件
	(2) pipeline.sh  src中的程序入口
	(3) combine_data.py  原始特征拼表
	(4) merge_and_split_data.py  拼接复赛和初赛数据，并且分割出验证集
	(5) build_uid2idx.py  特殊构造uid的idx，用频次来代替传统的LabelEncoding
	(6) DataLoader.py  数据读取类，作为入口的时候是用来构造特征的id
	(7) build_len_max_idx.py  统计长度特征的最大idx
	(8) build_pos_feature.py  构造与正样本相关的一些特征
	(9) build_conversion_rate.py  统计每种特征各个id的出现次数和转化率
	(10) main.py  训练和测试的入口
	(11) avg_submission.py  结果融合脚本
	(12) make_submission.py  构造提交格式的结果

---

## 特征介绍 ##
	我们团队使用了基础特征、长度特征、统计值特征、正样本统计值特征，下面做简单的介绍，具体的查看args.py
	(1) 基础特征
		- ad_static_features : 广告类的定长的基础特征
		- user_static_features : 用户类的定长的基础特征
		- user_dynamic_features : 用户类的不定长的基础特征
		- ignore_features : 忽略不适用的特征

	(2) 长度特性
		- user_dynamic_len_features : 对于不定长的特征，额外增加一个表示长度的特征
	
	(3) 统计值特征
		- uid : 因为uid比较稀疏，所以用统计的方法来建模uid
	
	(4) 正样本统计值特征
		- uid|ad_static_features : 因为单纯用统计的方法建模uid会有大量的信息损失，所以加入uid与ad拼接后的特征，并且统计对应的正样本出现次数，进一步建模uid
	
	在实际实验中，我们都构造了多版的特征子集，并不是用全部特征来训练和预测。
---

## 模型介绍 ##
	我们团队主要基于NFFM模型，针对不定长特征对基础模块修改了一下模型结构，而且也在模型结构上进行了一些创新。
	模型结构主要分三个模型：NFFM, NFFM_concat, NFFM_concat_triple
	(1) 基础模块
		- StaticEmbedding : 对定长的特征进行embedding
		- DynamicEmbedding : 对不定长的特征进行embedding，之后进行avg_pooling
	(2) 模型结构
		- NFFM : 传统NFFM模型结构
		- NFFM_concat : 在NFFM模型的基础上，把lr部分去掉，在bi-interaction层的输出层中，拼接上StaticEmbedding和DynamicEmbedding输出的embedding矩阵，作为NN层的输入。
		- NFFM_concat_triple : 在NFFM_concat的基础上，引入三阶性，对bi-interaction的输出再点乘上一个StaticEmbedding和DynamicEmbedding输出的embedding矩阵，作为NN层的输入。

---

## 结果融合 ##
	我们构造了多版的特征子集，在三个模型进行训练和验证，挑选验证集最好的前N个模型，进行按比例进行平均融合，得到最终结果。
