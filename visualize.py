'''
@ file function: 可视化工具，线下测试函数
@ author: 王中琦
@ data: 2022/3/20
'''

import json
import numpy as np
import torch
from models.softmax import softmax_net

from gensim.models.word2vec import Word2Vec

def main(sentence):

    device=torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu")
    net = softmax_net().to(device)
    net.load_state_dict(torch.load("results/18_softmax_net.pkl"))   # 导入训练好的模型


    words = sentence.split(" ")

    path = "count.json" # 导入one-hot编码
    with open(path, 'r', encoding='utf-8') as f:
        Info = json.load(f)

    # word_model = Word2Vec.load("my_word2vec_skip")

    for i in range(len(words)):
        feature1 = feature_judge(i-1,Info,len(words),words)
        feature2 = feature_judge(i,Info,len(words),words)
        feature3 = feature_judge(i+1,Info,len(words),words)

        feature_all = np.append(feature1,feature2)
        feature_all = np.append(feature_all,feature3)
        # try:
        #     feature_all =  word_model.wv[word.split('/')[0]]
        # except:
        #     feature_all = np.array(np.zeros((500)))

        feature = torch.from_numpy(feature_all)
        out = net(feature.to(torch.float32).to(device))
        result = out.argmax(dim=0).item()
        if result == 0:
            result = words[i]+"/B"
        elif result == 1:
            result = words[i]+"/I"
        else:
            result = words[i]+"/O"
        print(result)

def feature_judge(position,Info,lenth,words):
    '''
    [1,1,1,……]\n
    第一位表示句前\n
    第二位表示句后\n
    第三位表示未登陆\n
    '''
    lenth = lenth-1
    if position<0:  # 句前
        is_begin = np.array([1,0,1])
        feature = np.array(np.zeros((497)))
        feature = np.append(is_begin,feature)
    elif position>lenth:  # 句后
        is_end = np.array([0,1,1])
        feature = np.array(np.zeros((497)))
        feature = np.append(is_end,feature)      
    else:
        if Info.get(words[position].split('/')[0]) == None: # 未登陆
            is_now = np.array([0,0,1])
            feature = np.array(np.zeros((497)))
            feature = np.append(is_now,feature)
        else:
            is_now = np.array([0,0,0])
            feature = np.array(Info[words[position].split('/')[0]])
            feature = np.append(is_now,feature)
    return feature

if __name__ == "__main__":
    sentence = "江 泽民 , 李 鹏 , 乔 石 , 朱 镕基 , 李 瑞环 , 刘 华清 , 王 中琦 , 张 军 , 张 飞 , 叶 问 , 欧阳 修 , 陈 李 加菲 "
    main(sentence)

    # 结果如下：
    # 
