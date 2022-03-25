'''
@ file function: 计算每个词出现频率，统计前K个，将其转化为one-hot编码
@ author: 王中琦
@ data: 2022/3/14 
'''

from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json


def count_freq(words_list) -> dict:
    '''
    计算每个词出现频率
    '''
    words_list_hash = {} # 定义字典，方便hash查找
    words_len = len(words_list)
    for i in words_list:
        if(len(i)>=1): #对长度大于等于1的字/词做统计
            if words_list_hash.get(i) == None: # 如果这个词未在字典中再统计
                num = words_list.count(i) # 统计出现个数
                frq = num/words_len # 计算词频
                words_list_hash[i] = frq # 字典赋值
    return words_list_hash

def top_k(words_list_hash,k) -> dict:
    '''
    进行概率排序，选择最大出现概率
    '''
    num = 0
    top_k_list = {}
    # 以字典词频为关键字，从上到下排序
    a1 = sorted(words_list_hash.items(),key=lambda x:x[1],reverse=True)
    for i in a1:
        num+=1
        top_k_list[i[0]] = i[1]
        if(num == k): # 选取top-k个最高概率出现词语及对应词频
            break
    return top_k_list

def one_hot(top_k_list) -> dict:
    '''
    one-hot 编码
    '''
    data = []
    feature = {}
    for i in top_k_list:
        data.append(i)
    values = array(data) # 转换成numpy类型

    # 将其转换为数字（0~k-1）
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
 
    # 将其二值化
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # 转成字典记录
    for i in range(len(data)):
        feature[data[i]] = onehot_encoded[i]

    return feature


def count_feature(file_name) -> array:
    '''
    文件主函数，计算文本词语特征
    '''
    all_words = []

    with open(file_name,'r',encoding='utf-8') as inp:
        for line in inp.readlines(): # 读入每行文字
            line = line.rstrip("\n") # 去掉最后的换行符
            words = line.split(" ") # 以空格分割

            for word in words:
                try:
                    if word.split("/")[1] != 'O': # 只统计标签为'BI'的，防止无关词涌入太多
                        all_words.append(word.split("/")[0])
                except:
                    pass

    print("读入数据完毕")
    freq = count_freq(all_words)
    final_freq = top_k(freq,k=497)

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

    # 将最终one-hot保存到json文件中，方便日后读取
    operation = json.dumps(feature, cls=NumpyEncoder, ensure_ascii=False)
    f = open('count.json', 'w', encoding='utf-8')
    f.write(operation)
    f.close() # 注意用完及时关闭文件
