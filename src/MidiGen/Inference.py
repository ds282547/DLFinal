import platform

if platform.system() == 'Windows':
    isPlatformWindows = True
elif platform.system() == 'Linux':
    isPlatformWindows = False
else:
    print('This program does not support your platform!')
    quit()

if not isPlatformWindows:
    # for linux plotting
    import matplotlib as mpl

    mpl.use('Agg')

import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import pretty_midi
import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision.models
import torch.optim as optim
import argparse

import os
import platform
import random

import time

import matplotlib.pyplot as plt

from torch.autograd import Variable

import DataVisualization as dv


from RNN_class import *
# from DNNModel import *
from sklearn.metrics import f1_score


allEpoch = 20

inputFileMask = 'input_%d.npy'
outputFileMask = 'output_%d.npy'

import BeamSearch as bs

if isPlatformWindows:
    data_dir = 'E:\\Dataset\\AllTestingData\\'
    lossFiguresPath = '.\\figures\\'
elif platform.system() == 'Linux':
    data_dir = './Dataset/'
    lossFiguresPath = './figures/'

batch_size = 512
inputSizeW = 7
inputSizeH = 264
inputSize = inputSizeW * inputSizeH
outputLabelSize = 88

recordEveryBatch = 5
plotEverySong = 10

CNN_modelPath = 'CNN_models/model_epoch%d'
RNN_modelPath = '../RNN_models/model_epoch%d'
DNN_modelPath = '../DNN_models/model_epoch%d'
CNN_usedEpochModel = 13
RNN_usedEpochModel = 12
DNN_usedEpochModel = 13
useThreshold = False

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.cuda:
    device = torch.device('cuda')

if __name__ == '__main__':

    inputNode = inputSizeW * inputSizeH
    startTime = time.time()

    DNN_FScore = []
    RNN_FScore = []
    CNN_FScore = []

    N = 1184
    pairsN = 0
    filePairs = []

    # Collect input/output file pairs
    for i in range(1184):
        inputFileName = data_dir + (inputFileMask % i)
        outputFileName = data_dir + (outputFileMask % i)
        if os.path.isfile(inputFileName) and os.path.isfile(outputFileName):
            filePairs.append((inputFileName, outputFileName))

    pairsN = len(filePairs)
    print('Total file pairs = %d' % pairsN)

    # create directory for Loss Figure if path if it doesn't exist for file write
    if not os.path.exists(lossFiguresPath):
        os.makedirs(lossFiguresPath)
    # Show where to store the figure
    print('Figures path:%s' % lossFiguresPath)

    # inputVar = Variable(torch.zeros(batch_size, inputSize))
    # CNN_inputVar = Variable(torch.zeros(batch_size,inputSizeW, inputSizeH))
    # labelVar = Variable(torch.zeros(batch_size, outputLabelSize))
    # CNN_model = CNN.RNN()

    RNN_model = torch.load(RNN_modelPath % RNN_usedEpochModel)
    DNN_model = torch.load(DNN_modelPath % DNN_usedEpochModel)




    RNN_model.eval()
    DNN_model.eval()

    if args.cuda:
        # CNN_model = CNN_model.to(device)
        RNN_model = RNN_model.to(device)
        DNN_model = DNN_model.to(device)
        # inputVar = inputVar.to(device)
        # CNN_inputVar = CNN_inputVar.to(device)
        # labelVar = labelVar.to(device)

    pairIndex = 0
    losslist = []

    # random.shuffle(filePairs)

    for (inputFileName, outputFileName) in filePairs:
        torch.cuda.empty_cache()
        pairIndex += 1

        inputNP = np.load(inputFileName)
        labelNP = np.load(outputFileName)
        trainInfo = 'Song:{:3d}/{:3d}'.format(pairIndex, pairsN)

        inputTensor = torch.from_numpy(inputNP).float().to(device)
        # CNN_inputTensor = torch.from_numpy(inputNP).float().to(device)
        labelTensor = torch.from_numpy(labelNP).float().to(device)
        # print(inputTensor.shape)
        inputTensor = inputTensor.view(-1, inputSize)

        del labelNP

        # accum_loss = 0

        # CNN_output = torch.zeros(labelTensor.shape).to(device)

        RNN_output = torch.zeros(labelTensor.shape).to(device)
        # DNN_output = torch.zeros(labelTensor.shape).to(device)
        with torch.no_grad():
            RNN_output = RNN_model.forward(inputTensor)
            # DNN_output = DNN_model.forward(inputTensor)

        np.save('testopt',RNN_output.detach().cpu().numpy())
        exit()
        ##########################################################

        y_beams = bs.find_beam(RNN_output, 4)
        for t in range(len(RNN_output)):
            for k in range(4):
                yy = y_beams[k]
                l = np.log(bs.cal_prob(RNN_output[t].detach().cpu().numpy(),yy))       #something
                print(l)

        ##########################################################

        '''
        from CNN_class import *

        CNN_model = torch.load(CNN_modelPath % CNN_usedEpochModel)
        CNN_model.eval()
        if args.cuda:
            CNN_model = CNN_model.to(device)

        for i in range(inputNP.shape[0]):
            with torch.no_grad():
                CNN_output[i] = CNN_model.forward(CNN_inputTensor[i].unsqueeze(0))
        '''
        # loss = model.Loss(output, labelTensor)
        if useThreshold:
            # CNN_output = CNN_output > 0.5
            RNN_output = RNN_output > 0.5
            # DNN_output = DNN_output > 0.5

        # print("Input shape:{0}".format(inputTensor.shape))
        print("Ground-truth shape:{0}".format(labelTensor.shape))
        #print("Predict-label shape:{0}".format(output.shape))

        print(RNN_output[38032])
        exit()

        dv.showSongData3(inputNP, labelTensor,  RNN_output)

        # DNN_FScore.append(f1_score(labelTensor.detach().cpu().numpy(), DNN_output.detach().cpu().numpy(),average ='samples'))
        # CNN_FScore.append(f1_score(labelTensor.detach().cpu().numpy(), CNN_output.detach().cpu().numpy(),average ='samples'))
        # RNN_FScore.append(f1_score(labelTensor.detach().cpu().numpy(), RNN_output.detach().cpu().numpy(),average ='samples'))

        totalElapsedTime = time.time() - startTime
        print('Total training time : ' + time.strftime('%H:%M:%S', time.gmtime(totalElapsedTime)))
        # np.save("CNN_final", CNN_output.detach().cpu().numpy())
        # np.save("RNN_final", RNN_output.detach().cpu().numpy())
        # np.save("DNN_final", DNN_output.detach().cpu().numpy())
        # print(np.mean(DNN_FScore))
        print(np.mean(CNN_FScore))
        # print(np.mean(RNN_FScore))


    # np.save("DNN_FScore", DNN_FScore)
    np.save("CNN_FScore", CNN_FScore)
    # np.save("RNN_FScore", RNN_FScore)
    totalElapsedTime = time.time() - startTime
    print('Total training time : ' + time.strftime('%H:%M:%S', time.gmtime(totalElapsedTime)))

