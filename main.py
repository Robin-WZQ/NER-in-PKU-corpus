from datasets.Words_dataset import WordsDataset
from train_val_test import train,testall
from models.softmax import softmax_net
from torch.nn import init
import torch.nn
import torch
import torch.optim
import torch.utils
import warnings
warnings.filterwarnings("ignore")


def main():
    device=torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu")

    train_dataset = WordsDataset(function="name",dataset="train")
    test_dataset = WordsDataset(function="name",dataset="test")
    val_dataset = WordsDataset(function="name",dataset="val")

    net = softmax_net().to(device)

    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
        if 'bias' in name:
            init.constant_(param, val=1)

    optimizor= torch.optim.SGD(net.parameters(),lr=0.04,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoches = 50
    train_batch_size = 100
    test_batch_size = 100
    #建立一个数据迭代器
    # 装载训练集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=False)
    # 装载验证集
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=False)
    # 装载测试集
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=True)

    train(device,train_loader,optimizor,criterion,n_epoches,train_batch_size,net,val_loader)

    testall(device,test_loader,net)

    print("Done!")

if __name__ == "__main__":
    main()