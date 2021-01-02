## PyTorch model of the vanilla-networks used in SLIDE without LSH


import torch
import torch.nn as nn


class slideNet(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, out_feature_dim):
        super(slideNet, self).__init__()
        self.fc1 = nn.Linear(in_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_feature_dim)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def main():
    ## in_feature_dim, out_feature_dim taken from Amazon-670K
    in_feature_dim = 135909
    hidden_dim = 128
    batch_size = 256
    out_feature_dim = 670091
    inp = torch.randn([batch_size,in_feature_dim])
    net = slideNet(in_feature_dim, hidden_dim, out_feature_dim)

    out = net(inp)

    print(out.shape)


if __name__ == '__main__':
    main()
