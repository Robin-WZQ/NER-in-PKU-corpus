'''
@ file function: 主函数
@ author: 王中琦
@ data: 2022/3/17
'''

from datasets.Words_dataset import WordsDataset
from train_val_test import train,testall
from models.softmax import softmax_net
from torch.nn import init
import torch.nn
import torch
import torch.optim
import torch.utils
import warnings
import time
warnings.filterwarnings("ignore")


def main():
    # 自定义使用gpu or cpu
    device=torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu")

    n_epoches = 15 # 训练轮数
    train_batch_size = 50 # 训练batch
    val_batch_size = 100 # 验证batch
    test_batch_size = 100 # 测试batch

    # 导入数据
    train_dataset = WordsDataset(function="name",dataset="train")
    test_dataset = WordsDataset(function="name",dataset="test")
    val_dataset = WordsDataset(function="name",dataset="val")

    # 建立一个数据迭代器
    # 装载训练集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=False)
    # 装载验证集
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=val_batch_size,
                                            shuffle=False)
    # 装载测试集
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=True)
    # 导入模型至设备
    net = softmax_net().to(device)

    # 初始化模型参数
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01) # 权重使用正态分布
        if 'bias' in name:
            init.constant_(param, val=1) # 偏置使用常数

    optimizor= torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9) # 使用SGD优化器
    criterion = torch.nn.CrossEntropyLoss() # 交叉熵损失函数


    # 训练
    since = time.time()
    train(device,train_loader,optimizor,criterion,n_epoches,train_batch_size,net,val_loader)
    time_elapsed = time.time() - since
    print('训练结束！用时:{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 测试
    testall(device,test_loader,net)

    print("Done!")

if __name__ == "__main__":
    main()
