import os
import random

inputfileMask = 'input_%d.npy'
outputfileMask = 'input_%d.npy'
data_dir = 'D:\\processedMaestro\\'
N = 1184
filePairs = []
# Collect input/output file pairs

for i in range(1184):
    inputfile = data_dir + (inputfileMask % i)
    outputfile = data_dir + (outputfileMask % i)
    if os.path.isfile(inputfile) and os.path.isfile(outputfile):
        filePairs.append((inputfile, outputfile))


print('Total file pairs = %d' % len(filePairs))
random.shuffle(filePairs)