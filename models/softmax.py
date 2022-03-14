import torch
import torch.nn as nn
from torch.autograd import Variable


class softmax_net(nn.Module):
    """
    softmax network.
    """
    def __init__(self):
        super(softmax_net, self).__init__()
        self.layer = nn.Sequential(
                    nn.Linear(500,3),
                    # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == "__main__":
    input = Variable(torch.randn([100,500]))
    net =  softmax_net()
    output = net(input)
    print(output)
    #torch.Size([3])