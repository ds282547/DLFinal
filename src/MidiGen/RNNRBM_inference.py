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

import MidiGenNew.ToMidi as ToMidi
import MidiGenNew.BeamSearch as bs
import MidiGenNew.CalcYProb as caly
import queue


#ToMidi.ToMidi('E:\\Dataset\\AllTestingData\\output_901.npy','out_truth.midi')
#exit()

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

    def gibbs_prob(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\

        def gibbs_step(visible):

            # torch.matmul(visible, w) + bh
            h_probs = torch.sigmoid(F.linear(visible, w, bh))

            h_samples = self.sample_from_p(h_probs)

            # torch.matmul(h_samples, torch.t(w)) + bv

            v_probs = torch.sigmoid(F.linear(h_samples, w.t(), bv))

            #VSampler = distr.Binomial(total_count=1, probs=v_probs)
            v_samples = self.sample_from_p(v_probs)
            return h_probs, h_samples, v_probs, v_samples

        itorchuts = visible
        v_probs = None

        for k in range(num_steps):
            if k == 0:
                _, _, v_probs, v_sample = gibbs_step(itorchuts)
            else:
                _, _, v_probs, v_sample = gibbs_step(itorchuts)
            itorchuts = v_sample
        # torch.matmul(v_sample, w) + bh

        return v_probs

    def gibbs_sample(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\

        def gibbs_step(visible):

            # torch.matmul(visible, w) + bh
            h_probs = torch.sigmoid(F.linear(visible, w, bh))

            h_samples = self.sample_from_p(h_probs)

            # torch.matmul(h_samples, torch.t(w)) + bv

            v_probs = torch.sigmoid(F.linear(h_samples, w.t(), bv))

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
        # torch.matmul(v_sample, w) + bh
        h_probs1 = torch.sigmoid(F.linear(v_sample, w, bh))
        return v_sample, h_probs0, h_probs1

    def free_energy(self, v):
        # v (Vdim) vector  beacuse of RNN, BS = 1
        # bv (Vdim) vector
        # w (Hdim, Vdim) matrix
        v = v.view(1, -1)

        vbias_term = v.mv(self.bv)
        wx_b = F.linear(v, self.w, self.bh)
        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - vbias_term).mean()

    def forward_one(self, v_tm1, u_tm1 = None):

        def get_bias(u_tm1):
            bh_t = F.linear(u_tm1, self.wuh, self.bh)#torch.matmul(u_tm1, self.wuh) + self.bh
            bv_t = F.linear(u_tm1, self.wuv, self.bv)#torch.matmul(u_tm1, self.wuv) + self.bv
            return bh_t, bv_t


        if u_tm1 is None:
            u_tm1 = self.u0


        bh_t, bv_t = get_bias(u_tm1)
        ## gibbs sampling start from v_tm1
        v_t = self.gibbs_prob(v_tm1, self.w, bh_t, bv_t, num_steps=10)

        return v_t

    def get_rnn_hidden(self, v_t, u_tm1):
        activation = torch.matmul(self.wvu, v_t) + torch.matmul(self.wuu, u_tm1) + self.bu
        return torch.tanh(activation)

    def forward(self, visible, training=False):
        """
        forward pass for each sample (multiple steps)
        return list caches, which each element is
        the result from one timestep.
        """

        def get_bias(u_tm1):

            bh_t = F.linear(u_tm1, self.wuh, self.bh)#torch.matmul(u_tm1, self.wuh) + self.bh
            bv_t = F.linear(u_tm1, self.wuv, self.bv)#torch.matmul(u_tm1, self.wuv) + self.bv
            return bh_t, bv_t

        def get_rnn_hidden(v_t, u_tm1):
            activation = torch.matmul(self.wvu, v_t) + torch.matmul(self.wuu, u_tm1) + self.bu
            return torch.tanh(activation)

        time_steps = visible.shape[0]
        u_tm1 = self.u0
        # sum1 = 0.0
        # sum2 = 0.0

        total_cost = 0
        cost = 0
        mse = []
        for t in range(1, time_steps):
            v_t = visible[t]
            v_tm1 = visible[t - 1]
            bh_t, bv_t = get_bias(u_tm1)
            ## gibbs sampling start from v_tm1
            negative_sample, h_probs0, h_probs1 = self.gibbs_sample(v_tm1, self.w, bh_t, bv_t, num_steps=20)

            #mean_v = self.gibbs_step(negative_sample)[0]



            # RBM Loss
            cost += self.free_energy(v_t) - self.free_energy(negative_sample)
            if training:
                self.optimizer.zero_grad()
                cost.backward(retain_graph=(t != time_steps-1))
                self.optimizer.step()


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



            # um1 += v_t
            # sum2 += negative_sample
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

        self.lr = 0.001


    def sample_from_p(self, p):
        # CDK
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def get_rnn_hidden(self, v_t, u_tm1):
        activation = torch.matmul(self.wvu, v_t) + torch.matmul(self.wuu, u_tm1) + self.bu
        return torch.tanh(activation)


    def v_to_h(self, v, h_bias):
        p_h = F.sigmoid(F.linear(v, self.w, h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h, v_bias):
        p_v = F.sigmoid(F.linear(h, self.w.t(), v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def gibbs_prob(self, visible, w, bh, bv, num_steps=15):
        ## performace gibbs sampling for visible\

        def gibbs_step(visible):

            # torch.matmul(visible, w) + bh
            h_probs = torch.sigmoid(F.linear(visible, w, bh))

            h_samples = self.sample_from_p(h_probs)

            # torch.matmul(h_samples, torch.t(w)) + bv

            v_probs = torch.sigmoid(F.linear(h_samples, w.t(), bv))

            #VSampler = distr.Binomial(total_count=1, probs=v_probs)
            v_samples = self.sample_from_p(v_probs)
            return h_probs, h_samples, v_probs, v_samples

        itorchuts = visible
        v_probs = None

        for k in range(num_steps):
            if k == 0:
                _, _, v_probs, v_sample = gibbs_step(itorchuts)
            else:
                _, _, v_probs, v_sample = gibbs_step(itorchuts)
            itorchuts = v_sample
        # torch.matmul(v_sample, w) + bh

        return v_probs

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
        self.w += self.lr * self.dw
        self.bv += self.lr * self.dbv
        self.bh += self.lr * self.dbh
        self.wuv += self.lr * self.dwuv
        self.wuh += self.lr * self.dwuh
        # RNN grad
        self.bu += self.lr * self.dbu
        self.wvu += self.lr * self.dwvu
        self.wuu += self.lr * self.dwuu

    def forward_one(self, v_tm1, u_tm1 = None):

        def get_bias(u_tm1):
            bh_t = F.linear(u_tm1, self.wuh, self.bh)#torch.matmul(u_tm1, self.wuh) + self.bh
            bv_t = F.linear(u_tm1, self.wuv, self.bv)#torch.matmul(u_tm1, self.wuv) + self.bv
            return bh_t, bv_t


        if u_tm1 is None:
            u_tm1 = self.u0


        bh_t, bv_t = get_bias(u_tm1)
        ## gibbs sampling start from v_tm1
        v_t = self.gibbs_prob(v_tm1, self.w, bh_t, bv_t, num_steps=10)

        return v_t
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


            dbv_t = self.free_energy(n_v) - self.free_energy(v_t)



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





model = torch.load('RNNRBM_new_song72')
model = model

print(model.u0)
print(model.w)
print(model.wuv)
print(model.wuh)
print(model.wuu)
print(model.wvu)
print(model.bu)
print(model.bh)
print(model.bv)


acu = np.load('test_recording.npy')



acu = torch.from_numpy(acu)
print(acu.shape)

#ToMidi.ToMidiMatrixNew(acu,"test_acu_new.midi", threshold=0.1)
#exit()


N = acu.shape[0]



v0 = torch.zeros((88))




#model.forward_one(input0)



#for t in range(1, N):
    #piano[t] = vt
    #print(vt)


beam = queue.PriorityQueue()

log_yy_prob = 0
ut = model.u0
K = 4
K2 = K ** 4

beam.put((0, [], ut, log_yy_prob, v0))

Pyy = model.forward_one(v0, ut)


for t in range(0, 1066):
    print(t)
    new_beam = queue.PriorityQueue()


    x = acu[t].numpy()

    while not beam.empty():
        (l, s, ut, log_yy_prob, vyy) = beam.get()


        vt = model.forward_one(vyy, ut)
        torch.set_printoptions(precision=2)

        y_candidates = bs.find_beam2(acu[t].numpy(), vt.detach().numpy(), K)

        for k in range(K):
            yy = y_candidates[k]

            # P(yy, y[0:t-1])
            lang_pred = bs.cal_prob(vt, yy)
            # P(yy, x[t])
            acu_pred = bs.cal_prob(acu[t], yy)

            log_lang_pred = torch.log(lang_pred).sum().item()
            log_acu_prob = torch.log(acu_pred).sum().item()



            # log_accum  P(yy)
            # t=1, P(yy|y[0])+ P(yy}x[t]) - 0
            log_y_prob = caly.calLogProb(yy)
            ll = log_lang_pred + log_acu_prob - log_y_prob


            vyy = torch.zeros(88)
            vyy[yy] = 1

            ut = model.get_rnn_hidden(vyy, ut)

            new_beam.put((l+ll, s+[yy], ut, log_yy_prob, vyy))

    if new_beam.qsize() == K2:

        while new_beam.qsize() > K:
            new_beam.get()
        for t in new_beam.queue:
            print(t[0])

    beam = new_beam

l, s, _, _, _ = beam.get()
print(l)
piano = torch.zeros((len(s),88))
piano[0] = v0
for t in range(len(s)):
    print(t)
    print(s[t])
    piano[t, s[t]] = 1

ToMidi.ToMidiMatrixNew(piano.numpy(), 'Recording.midi', threshold=0.5)
