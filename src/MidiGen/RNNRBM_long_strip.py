import torch
import torch.nn as nn

import torch.distributions as distr
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt

import time

import math

import os

cuda = False
if cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')

class RNNRBM(nn.Module):

    def __init__(self, visible_dim = 2, hidden_dim = 50, rnn_hidden_dim = 50, scale=0.1):
        super(RNNRBM, self).__init__()

        scale = 1/ math.sqrt(hidden_dim)

        self.w = nn.Parameter(torch.randn((hidden_dim, visible_dim)) * scale)

        self.wuu = nn.Parameter(torch.randn((rnn_hidden_dim, rnn_hidden_dim)) * scale)
        self.wuv = nn.Parameter(torch.randn((visible_dim, rnn_hidden_dim)) * scale)
        self.wuh = nn.Parameter(torch.randn((hidden_dim, rnn_hidden_dim)) * scale)
        self.wvu = nn.Parameter(torch.randn((rnn_hidden_dim, visible_dim)) * scale)
        # bias of visable
        self.bv = nn.Parameter(torch.zeros(visible_dim))
        # bias of hidden
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        # bias of rnn hidden
        self.bu = nn.Parameter(torch.zeros(rnn_hidden_dim))
        self.u0 = nn.Parameter(torch.zeros(rnn_hidden_dim))
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.scale = scale

        self.optimizer = optim.SGD(self.parameters(),lr=1e-3)
        self.reg_factor = 0.2


    def sample_from_p(self, p):
        # CDK
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))



    def v_to_h(self, v, h_bias):
        p_h = F.sigmoid(F.linear(v, self.w, h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h, v_bias):
        p_v = F.sigmoid(F.linear(h, self.w.t(), v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v



    def gibbs_sample(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\
        '''
        def gibbs_step(visible):

            # torch.matmul(visible, w) + bh
            h_probs = torch.sigmoid(F.linear(visible, w, bh))

            h_samples = self.sample_from_p(h_probs)

            # torch.matmul(h_samples, torch.t(w)) + bv

            v_probs = torch.sigmoid(F.linear(h_samples, w.t(), bv))

            #VSampler = distr.Binomial(total_count=1, probs=v_probs)
            v_samples = self.sample_from_p(v_probs)
            return h_probs, h_samples, v_probs, v_samples
        '''

        #itorchuts = visible
        #h_probs0 = None

        v_ = visible
        for _ in range(num_steps):
            pre_h_, h_ = self.v_to_h(v_, bh)

            pre_v_, v_ = self.h_to_v(h_, bv)





        return visible, v_

    def free_energy(self, v):
        # v (Vdim) vector  beacuse of RNN, BS = 1
        # bv (Vdim) vector
        # w (Hdim, Vdim) matrix
        v = v.view(1, -1)

        vbias_term = v.mv(self.bv)
        wx_b = F.linear(v, self.w, self.bh)
        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - vbias_term).mean()

    def get_bias(self, u_tm1):

        bh_t = F.linear(u_tm1, self.wuh, self.bh)  # torch.matmul(u_tm1, self.wuh) + self.bh
        bv_t = F.linear(u_tm1, self.wuv, self.bv)  # torch.matmul(u_tm1, self.wuv) + self.bv

        return bh_t, bv_t

    def forward(self, visible, training=False):
        """
        forward pass for each sample (multiple steps)
        return list caches, which each element is
        the result from one timestep.
        """

        def get_rnn_hidden(v_t, u_tm1):
            activation = torch.matmul(self.wvu, v_t) + torch.matmul(self.wuu, u_tm1) + self.bu
            return torch.tanh(activation)

        time_steps = visible.shape[0]
        u_tm1 = self.u0
        sum1 = 0.0
        sum2 = 0.0

        total_cost = 0
        cost = 0
        mse = []
        for t in range(1, time_steps):
            v_t = visible[t]
            v_tm1 = visible[t - 1]
            #bh_t, bv_t = self.get_bias(u_tm1)
            bh_t = F.linear(u_tm1, self.wuh, self.bh)
            bv_t = F.linear(u_tm1, self.wuv, self.bv)

            ## gibbs sampling start from v_tm1
            # _, negative_sample = self.gibbs_sample(v_tm1, self.w, bh_t, bv_t, num_steps=20)

            v_ = v_t

            for _ in range(20):
                pre_h_, h_ = self.v_to_h(v_, bh_t)

                pre_v_, v_ = self.h_to_v(h_, bv_t)

            negative_sample = v_

            #mean_v = self.gibbs_step(negative_sample)[0]

            if training:
                self.optimizer.zero_grad()
                cost = self.free_energy(v_t) - self.free_energy(negative_sample)

                cost.backward(retain_graph=(t != time_steps-1))
                #print(self.u0.grad)
                self.optimizer.step()


            # RBM Loss
            # cost += self.free_energy(v_t) - self.free_energy(negative_sample)


            # RNN Loss
            y_t = torch.sigmoid(bv_t)
            total_cost += torch.sum(-v_t * torch.log(1e-6 + y_t) - (1 - v_t) * torch.log(1e-6 + 1 - y_t))

            '''

            cost = self.free_energy(v_t) - self.free_energy(negative_sample)

            if training:
                self.optimizer.zero_grad()
                cost.backward(retain_graph=(t != time_steps-1))


                self.optimizer.step()
            '''
            mse.append(torch.abs(v_t - negative_sample).mean().item())



            sum1 += v_t
            sum2 += negative_sample
            ut = get_rnn_hidden(v_t, u_tm1)
            u_tm1 = ut



        # regularization term
        total_cost /= time_steps
        reg_term = (torch.norm(self.wuv)+torch.norm(self.wuh) ) * self.reg_factor
        total_cost += reg_term

        return total_cost, mse,reg_term



class RNNRBM_NEW(nn.Module):

    def __init__(self, visible_dim = 2, hidden_dim = 50, rnn_hidden_dim = 50, scale=0.1):
        super(RNNRBM_NEW, self).__init__()

        scale = 1 / math.sqrt(hidden_dim)

        self.w = Variable(torch.randn((hidden_dim, visible_dim)) * scale, requires_grad=False)

        self.wuu = Variable(torch.randn((rnn_hidden_dim, rnn_hidden_dim)) * scale, requires_grad=False)
        self.wuv = Variable(torch.randn((visible_dim, rnn_hidden_dim)) * scale, requires_grad=False)
        self.wuh = Variable(torch.randn((hidden_dim, rnn_hidden_dim)) * scale, requires_grad=False)
        self.wvu = Variable(torch.randn((rnn_hidden_dim, visible_dim)) * scale, requires_grad=False)
        # bias of visable
        self.bv = Variable(torch.zeros(visible_dim), requires_grad=False)
        # bias of hidden
        self.bh = Variable(torch.zeros(hidden_dim), requires_grad=False)
        # bias of rnn hidden
        self.bu = Variable(torch.zeros(rnn_hidden_dim), requires_grad=False)
        self.u0 = Variable(torch.zeros(rnn_hidden_dim), requires_grad=False)
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.scale = scale



        self.dw = Variable(torch.zeros(self.w.shape), requires_grad=False)
        self.dbv = Variable(torch.zeros(self.bv.shape), requires_grad=False)
        self.dbh = Variable(torch.zeros(self.bh.shape), requires_grad=False)
        self.dwuv = Variable(torch.zeros(self.wuv.shape), requires_grad=False)
        self.dwuh = Variable(torch.zeros(self.wuh.shape), requires_grad=False)
        # RNN grad
        self.dbu = Variable(torch.zeros(self.bu.shape), requires_grad=False)
        self.dwvu = Variable(torch.zeros(self.wvu.shape), requires_grad=False)
        self.dwuu = Variable(torch.zeros(self.wuu.shape), requires_grad=False)

        self.lr = 0.00001


    def sample_from_p(self, p):
        # CDK
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))



    def v_to_h(self, v, h_bias):
        p_h = F.sigmoid(F.linear(v, self.w, h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h, v_bias):
        p_v = F.sigmoid(F.linear(h, self.w.t(), v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v



    def gibbs_sample(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\
        '''
        def gibbs_step(visible):

            # torch.matmul(visible, w) + bh
            h_probs = torch.sigmoid(F.linear(visible, w, bh))

            h_samples = self.sample_from_p(h_probs)

            # torch.matmul(h_samples, torch.t(w)) + bv

            v_probs = torch.sigmoid(F.linear(h_samples, w.t(), bv))

            #VSampler = distr.Binomial(total_count=1, probs=v_probs)
            v_samples = self.sample_from_p(v_probs)
            return h_probs, h_samples, v_probs, v_samples
        '''

        #itorchuts = visible
        #h_probs0 = None

        v_ = visible
        for _ in range(num_steps):
            pre_h_, h_ = self.v_to_h(v_, bh)

            pre_v_, v_ = self.h_to_v(h_, bv)




        return visible, v_

    def free_energy(self, v):
        # v (Vdim) vector  beacuse of RNN, BS = 1
        # bv (Vdim) vector
        # w (Hdim, Vdim) matrix
        v = v.view(1, -1)

        vbias_term = v.mv(self.bv)
        wx_b = F.linear(v, self.w, self.bh)
        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - vbias_term).mean()

    def get_bias(self, u_tm1):

        bh_t = F.linear(u_tm1, self.wuh, self.bh)  # torch.matmul(u_tm1, self.wuh) + self.bh
        bv_t = F.linear(u_tm1, self.wuv, self.bv)  # torch.matmul(u_tm1, self.wuv) + self.bv

        return bh_t, bv_t

    def updateGrad(self):
        self.w -= self.lr * self.dw
        self.bv -= self.lr * self.dbv
        self.bh -= self.lr * self.dbh
        self.wuv -= self.lr * self.dwuv
        self.wuh -= self.lr * self.dwuh
        # RNN grad
        self.bu -= self.lr * self.dbu
        self.wvu -= self.lr * self.dwvu
        self.wuu -= self.lr * self.dwuu

    def forward(self, visible, training=False):
        """
        forward pass for each sample (multiple steps)
        return list caches, which each element is
        the result from one timestep.
        """

        def get_rnn_hidden(v_t, u_tm1):
            activation = torch.matmul(self.wvu, v_t) + torch.matmul(self.wuu, u_tm1) + self.bu
            return torch.tanh(activation)

        time_steps = visible.shape[0]
        u_tm1 = self.u0

        all_ut = torch.zeros((time_steps, u_tm1.shape[0]))
        all_dbvt = torch.zeros((time_steps-1, self.bv.shape[0]))
        all_dbht = torch.zeros((time_steps-1, self.bh.shape[0]))

        all_ut[0] = u_tm1


        self.dw.zero_()
        self.dbv.zero_()
        self.dbh.zero_()
        self.dwuv.zero_()
        self.dwuh.zero_()

        self.dbu.zero_()
        self.dwvu.zero_()
        self.dwuu.zero_()


        mse = []
        for t in range(1, time_steps):
            v_t = visible[t]
            v_tm1 = visible[t - 1]
            #bh_t, bv_t = self.get_bias(u_tm1)
            bh_t = F.linear(u_tm1, self.wuh, self.bh)
            bv_t = F.linear(u_tm1, self.wuv, self.bv)

            _, n_v = self.gibbs_sample(v_tm1, self.w, bh_t, bv_t, num_steps=20)


            dbv_t = n_v - v_t

            #print('dbv_t {0}'.format(dbv_t))



            tempA = torch.sigmoid(torch.mv(self.w, n_v) - bh_t).unsqueeze(1)
            tempB = torch.sigmoid(torch.mv(self.w, v_t) - bh_t).unsqueeze(1)



            self.dw += torch.matmul( tempA , n_v.unsqueeze(0)) \
                    -  torch.matmul( tempB , v_t.unsqueeze(0))

            dbh_t = (tempA - tempB).squeeze(1)

            self.dbv += dbv_t
            self.dbh += dbh_t

            all_dbvt[t - 1] = dbv_t
            all_dbht[t - 1] = dbh_t
            all_ut[t] = u_tm1


            self.dwuh = torch.matmul(self.dbh.unsqueeze(1) , u_tm1.unsqueeze(0))
            self.dwuv = torch.matmul(self.dbv.unsqueeze(1) , u_tm1.unsqueeze(0))

            mse.append(torch.abs(v_t - n_v).mean().item())

            ut = get_rnn_hidden(v_t, u_tm1)

            '''

            cost = self.free_energy(v_t) - self.free_energy(n_v)

            if training:
                self.optimizer.zero_grad()
                cost.backward(retain_graph=(t != time_steps-1))


                self.optimizer.step()
            '''

            u_tm1 = ut

        #calc dbu, dwuu, dwvu by du_tp1
        # t = T -> 1
        du_tp1 = torch.zeros(u_tm1.shape) # in t=T

        for t in range(time_steps-2,0,-1):
            du_t = torch.mul(torch.mul(torch.mv(self.wuu,du_tp1),all_ut[t+1]), 1 - all_ut[t+1]) \
                 + torch.mv(self.wuh.t(), all_dbht[t]) \
                 + torch.mv(self.wuv.t(), all_dbvt[t])
            # t
            temp = torch.mul(torch.mul(du_t, all_ut[t]), 1 - all_ut[t])

            self.dbu += temp
            self.dwuu += torch.matmul(temp.unsqueeze(1), all_ut[t-1].unsqueeze(0))
            self.dwvu += torch.matmul(temp.unsqueeze(1), visible[t].unsqueeze(0))

        print('w:{0}'.format(torch.norm(self.w)))
        print('wuu:{0}'.format(torch.norm(self.wuu)))
        print('wuh:{0}'.format(torch.norm(self.wuh)))
        print('wuv:{0}'.format(torch.norm(self.wuv)))
        print('wvu:{0}'.format(torch.norm(self.wvu)))
        print('bh:{0}'.format(torch.norm(self.bh)))
        print('bu:{0}'.format(torch.norm(self.bu)))
        print('bv:{0}'.format(torch.norm(self.bv)))



        return mse




'''
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
        cost, mse = model(train_data[j:j+B], training = True)
        model.optimizer.zero_grad()
        cost.backward()
        model.optimizer.step()

        mse = np.array(mse)

    loz1.append(np.mean(mse))
    loz2.append(cost.item())
    end_time = time.time()
    print('epoch {:}, time {:.3f} secs'.format(i,end_time - start_time))


plt.plot(loz1)
plt.show()
plt.plot(loz2)
plt.show()
'''

allEpoch = 1
hidden_dim = 300
rnn_hidden_dim = 300
inputFileMask = 'output_%d.npy'
batch_size = 100
inputSize = 88

data_dir = 'D:\\processedMaestro\\'
lossFiguresPath = '.\\figures\\'


filePairs = []
# Collect input/output file pairs
for i in range(1184):
    inputFileName = data_dir + (inputFileMask % i)

    if os.path.isfile(inputFileName):
        filePairs.append(inputFileName)

pairsN = len(filePairs)
print('Total file pairs = %d' % pairsN)


model_old = torch.load('RNNRBM_epoch_small')
hidden_dim = model_old.hidden_dim
rnn_hidden_dim = model_old.rnn_hidden_dim


#print(hidden_dim)


model = RNNRBM_NEW(visible_dim=inputSize, hidden_dim=hidden_dim,rnn_hidden_dim=rnn_hidden_dim)

model.w = model_old.w.detach()
model.wvu = model_old.wvu.detach()
model.wuu = model_old.wuu.detach()
model.wuh = model_old.wuh.detach()
model.wuv = model_old.wuv.detach()
model.bu = model_old.bu.detach()
model.bh = model_old.bh.detach()
model.bv = model_old.bv.detach()


del model_old
inputVar = Variable(torch.zeros(batch_size, inputSize))



inputVar = Variable(torch.zeros(batch_size, inputSize))



global_costs = []
global_mses = []

for epoch in range(allEpoch):

    for j, inputFileName in enumerate(filePairs):
        inputNP = np.load(inputFileName)

        trainInfo = 'Song:{:3d}/{:3d}'.format(j, pairsN)

        inputTensor = torch.from_numpy(inputNP).float()
        #labelTensor = torch.from_numpy(labelNP).float().to(device)

        del inputNP
        #del labelNP

        winN = inputTensor.shape[0]

        mses = []

        total_batch = winN - batch_size
        i = 0
        while i+batch_size < winN:

            # print(inputVar)

            inputVar.data.copy_(inputTensor[i:i+batch_size])
            #labelVar.data.copy_(labelTensor[startIndex:endIndex])

            #model.optimizer.zero_grad()
            mse = model(inputVar, True)
            model.updateGrad()



            mse = np.array(mse).mean()


            mses.append(np.array(mse).mean())

            '''
            if i == 3000:
                print(output)
                print(target)
            '''
            if i % 3000 == 0:
                meanmse = np.array(mses).mean()
                global_mses.append(meanmse)

            print(trainInfo + ' Batch %3d/%3d Mse:%.5f ' % (i, total_batch,mse))
            i += 50

        '''
        remainInputVar = Variable(inputTensor[endIndex:winN]).to(device)
        remainBatchSize = winN - endIndex


        print(winN - endIndex)



        caches = []
        caches = model(inputVar, caches)

        # -log P(v(t))

        loss = -torch.mean(caches[-1][1])
        optimizer.zero_grad()
        loss.backward()
        cost = torch.mean(torch.abs(caches[-1][1] - remainInputVar[remainBatchSize  - 1])) / remainBatchSize
        costs.append(cost.cpu().item())
        '''




        np.save('RNNRBM_mses', global_mses)

        model.lr *= 0.99

        plt.figure()
        plt.plot(global_mses)
        plt.savefig('RNNRBM_costs_figure_song%d.png' % j)
        torch.save(model, 'RNNRBM_new_song%d' % j)