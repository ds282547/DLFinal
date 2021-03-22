import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
#import torchvision.models
import torch.optim as optim

class RNN(nn.Module):

    def __init__(self,inputNode):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size= 7 * 264,  # 图片每行的数据像素点
            hidden_size=200,  # rnn hidden unit
            num_layers=2,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(200, 88)  # 输出层

        self.Loss = torch.nn.BCELoss()
        self.learningRate = 0.01
        self.iterCount = 0
        self.optimizer = optim.SGD(self.parameters(), lr=self.learningRate, momentum=0.9)


    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        out = torch.nn.Sigmoid()(out)
        return out