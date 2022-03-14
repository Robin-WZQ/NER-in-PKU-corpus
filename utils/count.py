'''
@ file function: 计算每个词出现频率，统计前K个，将其转化为one-hot编码
@ author: 王中琦
@ data: 2022/3/14 
'''

from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json

def seg_word(all) -> list:
    '''
    去除停用词
    '''
    stopwords = set()
    with open('stopwords.txt','r',encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x :x not in stopwords,all))

def count_freq(words_list) -> dict:
    '''
    计算每个词出现频率
    '''
    words_list_hash = {}
    words_len = len(words_list)
    for i in words_list:
        if(len(i)>=1): #对长度大于等于1的字/词做统计
            if words_list_hash.get(i) == None: # 如果这个词未在字典中再统计
                num = words_list.count(i)
                frq = num/words_len
                words_list_hash[i] = frq
    return words_list_hash

def top_k(words_list_hash,k) -> dict:
    '''
    进行概率排序，选择最大出现概率
    '''
    num = 0
    top_k_list = {}
    a1 = sorted(words_list_hash.items(),key=lambda x:x[1],reverse=True)
    for i in a1:
        num+=1
        top_k_list[i[0]] = i[1]
        if(num == k):
            break
    return top_k_list

def one_hot(top_k_list) -> dict:
    data = []
    feature = {}
    for i in top_k_list:
        data.append(i)
    values = array(data)

    # 将其转换为数字（0~k-1）
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
 
    # 将其二值化
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    for i in range(len(data)):
        feature[data[i]] = onehot_encoded[i]

    return feature


def count_feature(file_name) -> array:
    '''
    文件主函数，计算文本词语特征
    '''
    all_words = []

    with open(file_name,'r',encoding='utf-8') as inp:
        for line in inp.readlines(): #读入每行文字
            line = line.rstrip("\n")
            words = line.split(" ")
            # words.pop()
            
            for word in words:
                try:
                    if word.split("/")[1] != 'O':
                        all_words.append(word.split("/")[0])
                except:
                    pass

    print("读入数据完毕")
    all_words = seg_word(all_words)
    freq = count_freq(all_words)
    final_freq = top_k(freq,k=500)

    feature = one_hot(final_freq)
    return feature

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types\n
    code from: https://stackoverflow.com/questions/57269741/typeerror-object-of-type-ndarray-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    file_name = "./data/train/name_train.txt"
    feature = count_feature(file_name)

    operation = json.dumps(feature, cls=NumpyEncoder, ensure_ascii=False)
    f = open('count.json', 'w', encoding='utf-8')
    f.write(operation)
    f.close()
