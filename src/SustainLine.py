import glob, os
import PreprocessFunc as prepFunc

import pickle
import numpy as np
import pretty_midi

import DataVisualization as dv
import matplotlib.pyplot as plt
import math


from scipy import interpolate


# wave/midi files path
d = 'D:\maestro-v1.0.0'


def get_piano_roll_mod(self, fs=100, times=None,
                   pedal_threshold=64):
    """Compute a piano roll matrix of this instrument.

    Parameters
    ----------
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    times : np.ndarray
        Times of the start of each column in the piano roll.
        Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
    pedal_threshold : int
        Value of control change 64 (sustain pedal) message that is less
        than this value is reflected as pedal-off.  Pedals will be
        reflected as elongation of notes in the piano roll.
        If None, then CC64 message is ignored.
        Default is 64.

    Returns
    -------
    piano_roll : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument.

    """
    # If there are no notes, return an empty matrix
    if self.notes == []:
        return np.array([[]] * 128)
    # Get the end time of the last event
    end_time = self.get_end_time()
    # Extend end time if one was provided
    if times is not None and times[-1] > end_time:
        end_time = times[-1]
    # Allocate a matrix of zeros - we will add in as we go
    piano_roll = np.zeros((128, int(fs * end_time)))
    # Drum tracks don't have pitch, so return a matrix of zeros
    if self.is_drum:
        if times is None:
            return piano_roll
        else:
            return np.zeros((128, times.shape[0]))
    # Add up piano roll matrix, note-by-note
    for note in self.notes:
        # Should interpolate
        startPt = int(note.start * fs)
        endPt = int(note.end * fs)
        length = endPt-startPt
        piano_roll[note.pitch,
        startPt:endPt] = np.arange(length,0,-1,dtype='float')/length


    # Process sustain pedals

    if pedal_threshold is not None:
        CC_SUSTAIN_PEDAL = 64
        time_pedal_on = 0
        is_pedal_on = False
        for cc in [_e for _e in self.control_changes
                   if _e.number == CC_SUSTAIN_PEDAL]:
            time_now = int(cc.time * fs)
            is_current_pedal_on = (cc.value >= pedal_threshold)
            if not is_pedal_on and is_current_pedal_on:
                time_pedal_on = time_now
                is_pedal_on = True
            elif is_pedal_on and not is_current_pedal_on:
                # For each pitch, a sustain pedal "retains"
                # the maximum velocity up to now due to
                # logarithmic nature of human loudness perception
                subpr = piano_roll[:, time_pedal_on:time_now]

                # Take the running maximum
                pedaled = np.maximum.accumulate(subpr, axis=1)
                piano_roll[:, time_pedal_on:time_now] = pedaled
                is_pedal_on = False

    '''
    # Process pitch changes
    # Need to sort the pitch bend list for the following to work
    ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
    # Add in a bend of 0 at the end of time
    end_bend = PitchBend(0, end_time)
    for start_bend, end_bend in zip(ordered_bends,
                                    ordered_bends[1:] + [end_bend]):
        # Piano roll is already generated with everything bend = 0
        if np.abs(start_bend.pitch) < 1:
            continue
        # Get integer and decimal part of bend amount
        start_pitch = pitch_bend_to_semitones(start_bend.pitch)
        bend_int = int(np.sign(start_pitch) * np.floor(np.abs(start_pitch)))
        bend_decimal = np.abs(start_pitch - bend_int)
        # Column indices effected by the bend
        bend_range = np.r_[int(start_bend.time * fs):int(end_bend.time * fs)]
        # Construct the bent part of the piano roll
        bent_roll = np.zeros(piano_roll[:, bend_range].shape)
        # Easiest to process differently depending on bend sign
        if start_bend.pitch >= 0:
            # First, pitch shift by the int amount
            if bend_int is not 0:
                bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
            else:
                bent_roll = piano_roll[:, bend_range]
            # Now, linear interpolate by the decimal place
            bent_roll[1:] = ((1 - bend_decimal) * bent_roll[1:] +
                             bend_decimal * bent_roll[:-1])
        else:
            # Same procedure as for positive bends
            if bend_int is not 0:
                bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
            else:
                bent_roll = piano_roll[:, bend_range]
            bent_roll[:-1] = ((1 - bend_decimal) * bent_roll[:-1] +
                              bend_decimal * bent_roll[1:])
        # Store bent portion back in piano roll
        piano_roll[:, bend_range] = bent_roll
    '''

    if times is None:
        return piano_roll
    piano_roll_integrated = np.zeros((128, times.shape[0]))
    # Convert to column indices
    times = np.array(np.round(times * fs), dtype=np.int)
    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if start < piano_roll.shape[1]:  # if start is >=, leave zeros
            if start == end:
                end = start + 1
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end],
                                                  axis=1)
    return piano_roll_integrated



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


index = 1
N = len(filePairs)
while index < N:
    pair = filePairs[index]

    print(pair[0])

    pmobj = pretty_midi.PrettyMIDI(pair[0])

    ins = pmobj.instruments[0]
    cclist = [_e for _e in ins.control_changes if _e.number == 64]


    origin_bins = math.ceil(pmobj.get_end_time() * prepFunc.sr / 512)

    ccdata = np.zeros(origin_bins)



    current_ccvalue = 0

    cc = cclist[0]
    if cc.time == 0:
        current_ccvalue = cc.value

    ccindex = 1
    cclength = len(cclist)

    for i in range(origin_bins):
        t = i * 0.032

        if t >= cc.time and ccindex < cclength:

            cc = cclist[ccindex]
            current_ccvalue = cc.value
            ccindex += 1

        ccdata[i] = current_ccvalue

    mytimes = np.arange(0, origin_bins, dtype="float") * (0.032)



    plt.figure()
    plt.plot(mytimes, ccdata)

    plt.title('Sustain signal of midi file')
    plt.show()


    #first interpolate
    '''

    f = interp1d(cctimes, ccvalues, kind='linear', fill_value='extrapolate')

    in_x = np.arange(0, pmobj.get_end_time(), 0.05)
    in_y = f(in_x)

    plt.plot(interpolated_x , interpolated_y)
    plt.show()

    # second interpolation

    origin_bins = math.ceil(pmobj.get_end_time() * prepFunc.sr / 512)
    mytimes = np.arange(0,origin_bins,dtype="float")*(0.032)

    
    f = interp1d(interpolated_x, interpolated_y, kind='cubic', fill_value='extrapolate')

    interpolated_x2 = np.arange(0, pmobj.get_end_time(), 0.032)
    interpolated_y2 = f(interpolated_x2)

    plt.plot(interpolated_x2 , interpolated_y2)
    '''


    plt.show()

    exit()

    piano_roll = pmobj.get_piano_roll(fs=100)

    print(piano_roll.shape)

    dv.showPiano(piano_roll)

    piano_roll = get_piano_roll_mod(ins,fs=100)

    print(piano_roll.shape)

    dv.showPiano(piano_roll)





    print('Processing File Pair %d/%d Name:%s' % (index+1, N, pair[1]))
    #(inputnp, times) = prepFunc.procWaveData(pair[1])



    #print("Input shape:{0}".format(inputnp.shape))

    #origin_bins = math.ceil(pmobj.get_end_time() * prepFunc.sr / 512)
    #mytimes = np.arange(0,origin_bins,dtype="float")*(0.032)


    exit()


    #outputnp = prepFunc.procMidiData(pmobj, times)
    #print("Output shape:{0}".format(outputnp.shape))
    #np.save(dtarget+('\\output_%d' % index), outputnp)

    #outputnp = None
    index += 1