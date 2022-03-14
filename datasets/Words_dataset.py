from distutils.log import info
import torch
import numpy as np
from torch.utils.data import Dataset
import json


class WordsDataset(Dataset):
    """
    This is the dataset function to get words and their label.\n

    Returns:
        word,label (numpy format) 

    Author: 王中琦
    """

    def __init__(self,function,dataset):
        file_name = "./data/"+dataset+"/"+str(function)+"_"+dataset+".txt"
        self.all_words = []
        path = "count.json"
        with open(path, 'r', encoding='utf-8') as f:
            Info = json.load(f)

        with open(file_name,'r',encoding='utf-8') as inp:
            for line in inp.readlines(): #读入每行文字
                words = line.split(" ")
                words.pop()
                for word in words:
                    if word.split("/")[1] == "B":
                        label = np.array(0)
                    elif word.split("/")[1] == "I":
                        label = np.array(1)
                    else:
                        label = np.array(2)

                    if Info.get(word.split('/')[0]) == None:
                        feature = np.array(np.zeros((500)))
                    else:
                        feature = np.array(Info[word.split('/')[0]])

                    label = torch.from_numpy(label)
                    feature = torch.from_numpy(feature)
                    
                    self.all_words.append((feature.to(torch.float32),label.to(torch.float32)))


    def __len__(self):
        return len(self.all_words)

    def __getitem__(self, index):
        # 导入数据到加载器.
        word,label = self.all_words[index]

        return (word,label)