import torch.nn as nn
import torch
import torch.optim as optim

class DNN(nn.Module):

    def __init__(self, inputNode):
        super(DNN, self).__init__()
        hiddenNode = 250
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(inputNode, hiddenNode),
            torch.nn.Dropout(0.3),  # drop 30% of the neuron
            #torch.nn.Sigmoid()
        )
        self.hidden2 = torch.nn.Sequential(
            torch.nn.Linear(hiddenNode, hiddenNode),
            torch.nn.Dropout(0.3),
            #torch.nn.Sigmoid()
        )
        self.hidden3 = torch.nn.Sequential(
            torch.nn.Linear(hiddenNode, hiddenNode),
            torch.nn.Dropout(0.3),
            #torch.nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(hiddenNode, 88),  # 输出层线性输出
            nn.Sigmoid()
        )
        #self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adadelta(self.parameters(), lr=0.01, rho=0.9, eps=1e-06, weight_decay=0)
        self.Loss = torch.nn.BCELoss()

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)

        return x