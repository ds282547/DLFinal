import torch
import torch.nn as nn

import torch.distributions as distr
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt

import time



class RNNRBM(nn.Module):

    def __init__(self, visible_dim = 2, hidden_dim = 50, rnn_hidden_dim = 50, scale=0.1):
        super(RNNRBM, self).__init__()

        self.w = nn.Parameter(torch.randn((visible_dim, hidden_dim)) * scale)

        self.wuu = nn.Parameter(torch.randn((rnn_hidden_dim, rnn_hidden_dim)) * scale)
        self.wuv = nn.Parameter(torch.randn((rnn_hidden_dim, visible_dim)) * scale)
        self.wuh = nn.Parameter(torch.randn((rnn_hidden_dim, hidden_dim)) * scale)
        self.wvu = nn.Parameter(torch.randn((visible_dim, rnn_hidden_dim)) * scale)
        # bias of visable
        self.bv = nn.Parameter(torch.zeros((1, visible_dim)))
        # bias of hidden
        self.bh = nn.Parameter(torch.zeros((1, hidden_dim)))
        # bias of rnn hidden
        self.bu = nn.Parameter(torch.zeros((1, rnn_hidden_dim)))
        self.u0 = nn.Parameter(torch.zeros((1, rnn_hidden_dim)))
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.scale = scale

        self.optimizer = optim.SGD(self.parameters(),lr=1e-3)


    def sample_from_p(self, p):
        # CDK
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def gibbs_sample(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\

        def gibbs_step(visible):

            h_probs = torch.sigmoid(torch.matmul(visible, w) + bh)

            #print('VV:{0} visiable : {1} HPOP {2}'.format(visible, torch.matmul(visible, w),h_probs))
            #HSampler = distr.Binomial(total_count=1, probs=h_probs)
            h_samples = self.sample_from_p(h_probs)

            #print('h sample {0}'.format(h_samples))

            v_probs = torch.sigmoid(torch.matmul(h_samples, torch.t(w)) + bv)

            #VSampler = distr.Binomial(total_count=1, probs=v_probs)
            v_samples = self.sample_from_p(v_probs)
            return h_probs, h_samples, v_probs, v_samples

        itorchuts = visible
        h_probs0 = None



        for k in range(num_steps):
            if k == 0:
                h_probs0, _, _, v_sample = gibbs_step(itorchuts)
            else:
                _, _, _, v_sample = gibbs_step(itorchuts)
            itorchuts = v_sample
        h_probs1 = torch.sigmoid(torch.matmul(v_sample, w) + bh)
        return v_sample, h_probs0, h_probs1

    def free_energy(self, v):
        # v (Vdim) vector  beacuse of RNN, BS = 1
        # bv (1, Vdim) matrix
        # w (Vdim, Hdim) matrix

        # unseq v to (1, Vdim)
        v = v.view(1,-1)

        # v(1, Vdim) * bv.T(Vdim, 1)  (inner product)
        vbias_term = torch.matmul(v, torch.t(self.bv) )


        #  v(1, Vdim) * w(Vdim, Hdim) = (1, Hdim)
        wx_b = torch.matmul(v.view(1,-1),self.w) + self.bh
        # wx_b (1, Hdim)
        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - vbias_term).mean()


    def forward(self, visible, training=False):
        """
        forward pass for each sample (multiple steps)
        return list caches, which each element is
        the result from one timestep.
        """

        def get_bias(u_tm1):

            bh_t = torch.matmul(u_tm1, self.wuh) + self.bh
            bv_t = torch.matmul(u_tm1, self.wuv) + self.bv
            return bh_t, bv_t

        def get_rnn_hidden(v_t, u_tm1):
            activation = torch.matmul(v_t, self.wvu) + torch.matmul(u_tm1, self.wuu) + self.bu
            return torch.tanh(activation)

        time_steps = visible.shape[0]
        u_tm1 = self.u0
        # sum1 = 0.0
        # sum2 = 0.0

        costz = []
        total_cost = 0
        mse = []
        for t in range(1, time_steps):
            v_t = visible[t]
            v_tm1 = visible[t - 1]
            bh_t, bv_t = get_bias(u_tm1)
            ## gibbs sampling start from v_tm1
            negative_sample, h_probs0, h_probs1 = self.gibbs_sample(v_tm1, self.w, bh_t, bv_t, num_steps=20)

            #mean_v = self.gibbs_step(negative_sample)[0]
            y_t = torch.sigmoid(bv_t)
            total_cost += torch.sum(-v_t*torch.log(1e-6+y_t) - (1-v_t)*torch.log(1e-6+1-y_t))

            '''
            cost = self.free_energy(v_t) - self.free_energy(negative_sample)

            if training:
                self.optimizer.zero_grad()
                cost.backward(retain_graph=(t != time_steps-1))
                self.optimizer.step()
            '''


            mse.append(torch.abs(v_t - negative_sample).mean().item())


            #costz.append(cost)

            # um1 += v_t
            # sum2 += negative_sample
            ut = get_rnn_hidden(v_t, u_tm1)
            u_tm1 = ut

        total_cost /= time_steps
        self.optimizer.zero_grad()
        total_cost.backward()
        print(self.wuu.grad)
        self.optimizer.step()
        return total_cost, mse #costz



model = RNNRBM()




N = 10
B = 4
train_data = torch.tensor([[0,0,0,0,0,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,0]])
train_data = torch.t(train_data)
train_data = train_data.type(torch.FloatTensor)

loz1 = []
loz2 = []
for i in range(10000):

    start_time = time.time()
    for j in range(N - B):
        costss, mse = model(train_data[j:j+B], training = True)


        mse = np.array(mse)
        loz1.append(np.mean(mse))
        loz2.append(costss.item())
    end_time = time.time()
    print('epoch {:}, time {:.3f} secs'.format(i,end_time - start_time))

plt.plot(loz1)
plt.show()
plt.plot(loz2)
plt.show()
