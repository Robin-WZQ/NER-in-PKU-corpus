# NER-in-PKU-corpus
北理工知识工程实验一：基于北大中文语料（1998）的命名实体识别

## Code Tree

|data

-----|train #训练集

-----|test #测试集

-----|val #验证集

|datasets

-----|Words_dataset.py #数据集文件，包括读入文件+预处理

|models

-----|softmax.py #模型

|results

-----|{epoch}_softmax_net.pkl #训练好的模型

|runs

-----|some log file (by tensorboard file) #记录训练过程的文件

|utils

-----|count.py # 统计词频，并赋予one-hot向量，会生成中间处理的json文件

-----|label2BIO.py #预处理文件，将源人民日报标注文件，转化为相应的BIO文件

main.py # 主文件

count.json #数据向量表示文件，字典->json

train_val_test.py #训练+验证+测试相关代码

visualize.py #可视化函数，给定一段文字，用训练好的模型进行识别


## Usage

1. 安装相关依赖

```
pip install -r requirements.txt
```

2. 源文件预处理

run utils/label2BIO.py

3. 词频统计+one-hot

run count,py，会生成count.json文件

4. 训练

run main.py

5. 线下推断

run visualize.py
