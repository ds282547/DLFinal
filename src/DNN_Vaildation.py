import platform

import sklearn.metrics as skm

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
#import librosa
#import librosa.display
#import matplotlib.pyplot as plt
#import pretty_midi
import torch
import torch.nn.functional as F
import torch.nn as nn
#import torchvision.models
import torch.optim as optim
import argparse

import os
import platform
import random

import time


import matplotlib.pyplot as plt

from torch.autograd import Variable

from RNN_class import *

import DataVisualization as dv


allEpoch = 20

inputFileMask = 'input_%d.npy'
outputFileMask = 'output_%d.npy'

if isPlatformWindows:
    data_dir = 'D:\\processedMaestro\\'
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

usedEpochModel = 0

usingThreshold = False

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

    inputVar = Variable(torch.zeros(batch_size, inputSize))
    labelVar = Variable(torch.zeros(batch_size, outputLabelSize))


    model = torch.load('DNN_Model/model_epoch%d' % usedEpochModel)

    model.eval()

    print(model)

    if args.cuda:
        model = model.to(device)
        inputVar = inputVar.to(device)
        labelVar = labelVar.to(device)






    pairIndex = 0
    losslist = []

    fscorelist = []

    random.shuffle(filePairs)

    for (inputFileName, outputFileName) in filePairs:

        pairIndex += 1

        inputNP = np.load(inputFileName)
        labelNP = np.load(outputFileName)
        trainInfo = 'Song:{:3d}/{:3d}'.format(pairIndex,pairsN)


        inputTensor = torch.from_numpy(inputNP).float().to(device)
        labelTensor = torch.from_numpy(labelNP).float().to(device)

        inputTensor = inputTensor.view(-1, inputSize)

        del labelNP

        accum_loss = 0

        output = torch.zeros(labelTensor.shape).to(device)

        #for i in range(500):
        #    output[i] = model.forward(inputTensor[i])
        output = model.forward(inputTensor)



        print("Input shape:{0}".format(inputTensor.shape))
        print("Ground-truth shape:{0}".format(labelTensor.shape))
        print("Predict-label shape:{0}".format(output.shape))


        '''
        if usingThreshold:
            output = output > 0.5
            fscore = skm.f1_score(labelTensor.cpu().numpy(), output.cpu().detach().numpy(), average='samples')
            print(fscore)
            fscorelist.append(fscore)
        '''
        if usingThreshold:
            output = output > 0.5



        dv.showSongData2(inputNP,labelTensor,output)


        '''
        
        if pairIndex % plotEverySong == 0 and pairIndex > 0:
            plt.plot(losslist)
            plt.savefig(lossFiguresPath+'result_%d.png' % pairIndex)
        '''

        totalElapsedTime = time.time() - startTime
        print('Total training time : '+ time.strftime('%H:%M:%S', time.gmtime(totalElapsedTime)))



    plt.plot(fscorelist)
    plt.show()


    totalElapsedTime = time.time() - startTime
    print('Total training time : '+ time.strftime('%H:%M:%S', time.gmtime(totalElapsedTime)))

