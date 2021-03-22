import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

def showSongData(input, label, output):
    ip = input
    ip = np.average(ip, axis=1)
    lb = label.cpu().numpy()
    op = output.cpu().detach().numpy()

    ip = ip.T
    lb = lb.T
    op = op.T

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)



    cax = ax1.matshow(ip, interpolation='nearest', aspect='auto')

    cax = ax2.matshow(op, interpolation='nearest', aspect='auto')

    cax = ax3.matshow(lb, interpolation='nearest', aspect='auto')

    plt.show()

def showSongData2(input, label, output, cmap='magma'):
    ip = input
    ip = np.average(ip, axis=1)
    lb = label.cpu().numpy()
    op = output.cpu().detach().numpy()

    ip = ip.T
    lb = lb.T
    op = op.T

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)


    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    im1 = ax1.imshow(ip,cmap='gist_stern', interpolation='nearest', aspect='auto', origin='lower')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = ax2.imshow(op,cmap=cmap, interpolation='nearest', aspect='auto', origin='lower')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im3 = ax3.imshow(lb,cmap=cmap, interpolation='nearest', aspect='auto', origin='lower')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')



    plt.show()

def showSongDataWithModels(input, label, models_outputs):
    ip = input
    ip = np.average(ip, axis=1)
    lb = label.cpu().numpy()


    ip = ip.T
    lb = lb.T



    fig = plt.figure()

    N = (len(models_outputs)+2)*100

    ax1 = fig.add_subplot(N + 11)
    ax2 = fig.add_subplot(N + 12)


    cax = ax1.matshow(ip, interpolation='nearest', aspect='auto')
    cax = ax2.matshow(lb, interpolation='nearest', aspect='auto')

    index = 3
    for output in models_outputs:
        op = output.cpu().detach().numpy()
        op = op.T
        axz = fig.add_subplot(N + 10 + index)
        cax = axz.matshow(op, interpolation='nearest', aspect='auto')

        index += 1

    plt.show()

def showPiano(matrix):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cax = ax1.matshow(matrix, interpolation='nearest', aspect='auto')

    plt.show()