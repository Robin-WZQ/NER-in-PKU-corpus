import torch
import numpy as np
from torch.utils.data import Dataset
import json
from gensim.models.word2vec import Word2Vec


class WordsDataset(Dataset):
    """
    This is the dataset function to get words and their label.\n

    Returns:
        word,label (numpy format) 

    Author: 王中琦
    """

    def __init__(self,function,dataset):
        file_name = "./data/"+dataset+"/"+str(function)+"_"+dataset+".txt" #获取文件名称
        self.all_words = []
        path = "count.json" # 计算好的one-hot编码
        # word_model = Word2Vec.load("my_word2vec_skip")
        with open(path, 'r', encoding='utf-8') as f:
            Info = json.load(f) # 读入计算好的one-hot编码

        def feature_judge(position,Info,lenth,words) -> np:
                '''
                [1,1,1,……]\n
                第一位表示句前\n
                第二位表示句后\n
                第三位表示未登陆\n
                '''
                lenth = lenth-1
                if position<0: # 句前
                    is_begin = np.array([1,0,1])
                    feature = np.array(np.zeros((497)))
                    feature = np.append(is_begin,feature)
                elif position>lenth: # 句后
                    is_end = np.array([0,1,1])
                    feature = np.array(np.zeros((497)))
                    feature = np.append(is_end,feature)      
                else:
                    if Info.get(words[position].split('/')[0]) == None: #未登陆
                        is_now = np.array([0,0,1])
                        feature = np.array(np.zeros((497)))
                        feature = np.append(is_now,feature)
                    else:
                        is_now = np.array([0,0,0])
                        feature = np.array(Info[words[position].split('/')[0]])
                        feature = np.append(is_now,feature)
                return feature


        with open(file_name,'r',encoding='utf-8') as inp:
            for line in inp.readlines(): #读入每行文字
                words = line.split(" ") # 以空格分割
                words.pop() # 去掉换行符
                # 将BIO转换成'012'数字表示
                for i in range(len(words)):
                    if words[i].split("/")[1] == "B":
                        label = np.array(0) 
                    elif words[i].split("/")[1] == "I":
                        label = np.array(1)
                    else:
                        label = np.array(2)
                    
                    
                    feature1 = feature_judge(i-1,Info,len(words),words) # 计算目标词前一个词特征
                    feature2 = feature_judge(i,Info,len(words),words) # 计算目标词特征
                    feature3 = feature_judge(i+1,Info,len(words),words) # 计算目标词前一个词特征

                    feature_all = np.append(feature1,feature2) # 向量按行拼接
                    feature_all = np.append(feature_all,feature3) # 向量按行拼接
                    

                    # try:
                    #     feature_all =  word_model.wv[word.split('/')[0]]
                    # except:
                    #     feature_all = np.array(np.zeros((500)))

                    label = torch.from_numpy(label) # 将标签转换成tensor类型
                    feature = torch.from_numpy(feature_all) # 将特征值转换成tensor类型
                    
                    self.all_words.append((feature.to(torch.float32),label.to(torch.float32)))


    def __len__(self):
        # 返回长度
        return len(self.all_words)

    def __getitem__(self, index) -> tuple:
        # 导入数据到加载器.
        word,label = self.all_words[index]

        return (word,label)

    
