# 命名实体识别实验——代码说明


## Background - 背景

> 北京理工大学知识工程实验一：完成基于北大中文预料（1998）的命名实体识别任务。具体的，分别针对人名、地名以及机构名对预料进行BIO标注，统计词频并选择top-k个进行one-hot编码。设计3个3分类器，能够对编码后文本词语进行softmax分类，最终输出BIO类别。

## Usage - 使用方法

1. 安装相关依赖

```
pip install -r requirements.txt
```

2. 源文件预处理

```
python utils/label2BIO.py
```

3. 词频统计+one-hot

```
python utils/count,py
```

最终会产生一个中间文件count.json，存储着top-k个词语的one-hot向量表示。

4. 训练

```
python main.py
```

5. 线下推断

```
python visualize.py
```

对于任意一个给定的，已经分好词的语句，进行线下可视化推断。

## Code Tree - 代码树说明

|data

-----|train # 训练集

-----|test # 测试集

-----|val # 验证集

|datasets

-----|Words_dataset.py # 数据集文件，包括读入文件+预处理

|models

-----|softmax.py # 模型

|results

-----|{epoch}_softmax_net.pkl # 训练好的模型

|runs

-----|some log file (by tensorboard file) # 记录训练过程的文件

|utils

-----|count.py # 统计词频，并赋予one-hot向量，会生成中间处理的json文件

-----|label2BIO.py # 预处理文件，将源人民日报标注文件，转化为相应的BIO文件

-----|Wordvector.py # 利用word2vector先进行word imbedding，得到k(默认为500)维向量

main.py # 主文件

count.json # 数据向量表示文件，字典->json

train_val_test.py # 训练+验证+测试相关代码

visualize.py # 可视化函数，给定一段文字，用训练好的模型进行识别

stopwords.txt # 停词，在计算word2vector时使用的

## RF
https://github.com/Robin-WZQ/BIT-AI-Review
