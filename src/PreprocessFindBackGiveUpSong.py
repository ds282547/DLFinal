import glob, os
import PreprocessFunc as prepFunc

import pickle
import numpy as np
import pretty_midi

# wave/midi files path
d = 'D:\maestro-v1.0.0'
# target dir
dtarget = 'D:\processedMaestro'


subdirs = [os.path.join(d, o) for o in os.listdir(d)
                    if os.path.isdir(os.path.join(d,o))]
print('There are {1} folder(s) in \'{0}\'.'.format(d,len(subdirs)))

filePairs = []
filenameList = []

for subdir in subdirs:
    for midiFile in glob.glob(subdir+'\\*.midi'):

        onlyFileName = os.path.splitext(midiFile)[0]
        wavFile = onlyFileName  + '.wav'
        filenameList.append(onlyFileName)

        if os.path.isfile(wavFile):
            filePairs.append((midiFile,wavFile))

print("Total Wav-Midi File Pairs: %d " % len(filePairs))

# create directory if path if it doesn't exist for file write
if not os.path.exists(dtarget):
    os.makedirs(dtarget)

index = 1
N = len(filePairs)
while index < N:
    pair = filePairs[index]

    pmobj = pretty_midi.PrettyMIDI(pair[0])

    if pmobj.get_end_time()*prepFunc.sr < 32000000:

        pmobj = None
        index += 1
        continue

    print('Find back Give-up song..')

    print('Processing File Pair %d/%d Name:%s' % (index+1, N, pair[1]))




    (inputnp, times) = prepFunc.procWaveData(pair[1])


    print("Input shape:{0}".format(inputnp.shape))
    np.save(dtarget+('\\input_%d' % index), inputnp)
    inputnp = None



    outputnp = prepFunc.procMidiData(pmobj, times)
    print("Output shape:{0}".format(outputnp.shape))
    np.save(dtarget+('\\output_%d' % index), outputnp)

    outputnp = None

    index += 1


# save filename list
with open(dtarget+'\\filename_list.list', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(filenameList, filehandle)

