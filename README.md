# 基于BERT的情感分析模块。

**基于[别人的代码](https://blog.csdn.net/Real_Brilliant/article/details/84880528).**

**BERT参考：https://github.com/huggingface/pytorch-pretrained-BERT**

`pytorch-pretrained-BERT`是一个基于`Pytorch`的封装好的BERT框架，使用其中的`BertForSequenceClassification`。

训练：

数据格式：

每一个行为一个json字符串，json字符串内有`label`、`text`字段。

`label`字段与train.py的`MyPro.get_labels()`对应。


数据格式：json list,每一个元素长度不超过120字符，超过后自动截断，list长度不超过32.

返回数据：json list,顺序对应请求的list顺序，包含`label`、`scores`字段

训练数据集来源：[https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb)
