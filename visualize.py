import jieba
import json
import numpy as np
import torch
from models.softmax import softmax_net
device=torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu")
net = softmax_net().to(device)
net.load_state_dict(torch.load("results/48_softmax_net.pkl"))


sentence = "江 泽民 , 李 鹏 , 乔 石 , 朱 镕基 , 李 瑞环 , 刘 华清"
# seg_list = jieba.cut(sentence)
seg_list = sentence.split(" ")
feature_list = []

path = "count.json"
with open(path, 'r', encoding='utf-8') as f:
    Info = json.load(f)

for word in seg_list:
    if Info.get(word) == None:
        feature = np.array(np.zeros((500)))
    else:
        feature = np.array(Info[word])

    feature = torch.from_numpy(feature)
    out = net(feature.to(torch.float32).to(device))
    result = out.argmax(dim=0).item()
    if result == 0:
        result = word+"/B"
    elif result == 1:
        result = word+"/I"
    else:
        result = word+"/O"
    print(result)

