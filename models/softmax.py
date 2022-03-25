import torch
import torch.nn as nn
from torch.autograd import Variable


class softmax_net(nn.Module):
    """
    softmax network.
    """
    def __init__(self):
        super(softmax_net, self).__init__()
        # 使用Sequential容器进行封装
        self.layer = nn.Sequential(
                    nn.Linear(1500,3),
                    # softmax已经定义在了torch中的CrossEntropyLoss中，所以这里不用编写
        )

    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == "__main__":
    input = Variable(torch.randn([100,1500]))
    net =  softmax_net()
    output = net(input)
    print(output)
    #torch.Size([3])
