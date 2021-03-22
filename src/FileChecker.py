import os
import numpy as np
from pypianoroll import Multitrack, Track
import matplotlib.pyplot as plt

inputPath = 'D:\\processedMaestro\\input_%d.npy'
outputPath = 'D:\\processedMaestro\\output_%d.npy'

filePairs = []

for i in range(1184):
    inputFullPath = inputPath % i
    outputFullPath = outputPath % i
    if not os.path.isfile(inputFullPath):
        continue
    if not os.path.isfile(outputFullPath):
        continue
    filePairs.append((inputFullPath, outputFullPath))

'''
maxlist = []
for i in range(len(filePairs)):
    pair = filePairs[i]
    ip = np.load(pair[0])
    npm = np.max(ip)
    print(npm)
    maxlist.append(npm)

plt.plot(maxlist)
plt.show()
exit()
'''

ip = np.load(pair[0])
op = np.load(pair[1])

ip = np.average(ip, axis=1)

ip = ip.T
op = op.T

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

cax = ax1.matshow(ip[:,560:680], interpolation='nearest', aspect='auto')

cax = ax2.matshow(op[:,560:680], interpolation='nearest', aspect='auto')





plt.show()
'''
pr = np.load(outputPath % 4)
print(pr.shape)

k = np.zeros((44079,128))
k[:,0:88] = pr
pr = None
track = Track(pianoroll=k, program=0, is_drum=False,
              name='my awesome piano')

# Plot the piano-roll
fig, ax = track.plot()
plt.show()
'''