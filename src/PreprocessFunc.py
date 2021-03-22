import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from pretty_midi.containers import PitchBend
from pretty_midi.utilities import pitch_bend_to_semitones
# filename = "./MIDI-Unprocessed_Track01_wav.wav"
# y, sr = librosa.load('./MIDI-Unprocessed_Track01_wav.wav')

from pypianoroll import Multitrack, Track

sr = 16000  # sample/s
hop_length = 512
window_size = 7
min_midi = 21
max_midi = 108


def wav2inputnp(audio_fn,spec_type='cqt',bin_multiple=3):

    bins_per_octave = 12 * bin_multiple #should be a multiple of 12
    n_bins = (max_midi - min_midi + 1) * bin_multiple

    #down-sample,mono-channel
    y,_ = librosa.load(audio_fn,sr)
    print("Original shape:", y.shape)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins)



    print("After cqt(shape):",S.shape)
    S = S.T
    S = np.abs(S)

    minDB = np.min(S)

    #print(np.min(S),np.max(S),np.mean(S))

    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)

    windows = []

    # IMPORTANT NOTE:
    # Since we pad the the spectrogram frame,
    # the onset frames are actually `offset` frames.
    # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
    # starting at frame 0 of the padded spectrogram
    for i in range(S.shape[0]-window_size+1):
        w = S[i:i+window_size,:]
        windows.append(w)

    #print inputs
    x = np.array(windows)

    return x

def fixed_get_piano_roll(self, fs=100, times=None,
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

    # Get the end time of the last event
    end_time = self.get_end_time()
    totalfstime = int(fs*end_time)

    # Extend end time if one was provided
    if times is not None and times[-1] > end_time:
        end_time = times[-1]
    # Allocate a matrix of zeros - we will add in as we go


    # Process pitch changes
    # Need to sort the pitch bend list for the following to work

    # Process sustain pedals
    if pedal_threshold is not None:
        CC_SUSTAIN_PEDAL = 64
        time_pedal_on = 0
        is_pedal_on = False
        cclist = [_e for _e in self.control_changes
                   if _e.number == CC_SUSTAIN_PEDAL]
        ccount = len(cclist)


    piano_roll_integrated = np.zeros((128, times.shape[0]))

    times = np.array(np.round(times * fs), dtype=np.int)

    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):

        if start < totalfstime:  # if start is >=, leave zeros
            if start == end:
                end = start + 1

            winstartpt = start
            winendpt = end
            windowsiz = winendpt - winstartpt

            print("Win(%d,%d)" % (winstartpt, winendpt), end='')

            piano_roll = np.zeros((128, windowsiz))

            notecount = 0
            for note in self.notes:
                # Should interpolate
                notestartpt = int(note.start * fs)
                noteendpt = int(note.end * fs)



                #Note exceed window/frame range
                if notestartpt >= winendpt:
                    break

                notestartpt -=  winstartpt
                noteendpt -= winstartpt

                if noteendpt <= 0:
                    continue

                notestartpt = max(notestartpt, 0)
                noteendpt = min(windowsiz, noteendpt)

                piano_roll[note.pitch,
                notestartpt:noteendpt] += note.velocity

                notecount += 1
            if pedal_threshold is not None:
                time_pedal_on = 0
                is_pedal_on = False

                for cc in cclist:
                    time_now = int(cc.time * fs)

                    if time_now >= winendpt:
                        break

                    is_current_pedal_on = (cc.value >= pedal_threshold)
                    if not is_pedal_on and is_current_pedal_on:
                        time_pedal_on = time_now
                        is_pedal_on = True
                    elif is_pedal_on and not is_current_pedal_on:
                        # For each pitch, a sustain pedal "retains"
                        # the maximum velocity up to now due to
                        # logarithmic nature of human loudness perception
                        safewinleft = max(0, time_pedal_on - winstartpt)
                        safewinright = min(windowsiz, time_now - winstartpt)



                        subpr = piano_roll[:, safewinleft:safewinright]

                        # Take the running maximum
                        pedaled = np.maximum.accumulate(subpr, axis=1)
                        saftwinleft = max(0, time_pedal_on - winstartpt)
                        saftwinright = min(windowsiz, time_now - winstartpt)
                        piano_roll[:, safewinleft:safewinright] = pedaled

                        is_pedal_on = False

            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll,
                                                          axis=1)

            print(" #%d Notecount = %d" % (n,notecount))


    # Drum tracks don't have pitch, so return a matrix of zeros
    if self.is_drum:
        if times is None:
            return piano_roll
        else:
            return np.zeros((128, times.shape[0]))
    # Add up piano roll matrix, note-by-note



    # Convert to column indices
    """
    times = np.array(np.round(times * fs), dtype=np.int)
    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if start < piano_roll.shape[1]:  # if start is >=, leave zeros
            if start == end:
                end = start + 1
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end],
                                                  axis=1)
    """
    return piano_roll_integrated

def pm_get_piano_roll(pm, fs=100, times=None, pedal_threshold=64):
    """Compute a piano roll matrix of the MIDI data.
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
        Piano roll of MIDI data, flattened across instruments.
    """

    # If there are no instruments, return an empty array
    if len(pm.instruments) == 0:
        return np.zeros((128, 0))

    # Get piano rolls for each instrument
    piano_rolls = [fixed_get_piano_roll(self=i,fs=fs, times=times,
                                    pedal_threshold=pedal_threshold)
                   for i in pm.instruments]
    # Allocate piano roll,
    # number of columns is max of # of columns in all piano rolls
    piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
    # Sum each piano roll into the aggregate piano roll
    for roll in piano_rolls:
        piano_roll[:, :roll.shape[1]] += roll
    return piano_roll


def mid2outputnp(pm_mid,times):


    piano_roll = pm_mid.get_piano_roll(fs=300,times=times)

    piano_roll = piano_roll[min_midi:max_midi+1].T
    piano_roll[piano_roll > 0] = 1

    return piano_roll

'''
pm_mid = pretty_midi.PrettyMIDI("MIDI-Unprocessed_Track01_wav.midi")
inputnp = wav2inputnp(filename)
times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=sr,hop_length=hop_length)
outputnp = mid2outputnp(pm_mid,times)
print(outputnp.shape)
np.save("input",inputnp)
np.save("output",outputnp)
'''

def procWaveData(wavefname):
    inputnp = wav2inputnp(wavefname)
    times = librosa.frames_to_time(np.arange(inputnp.shape[0]), sr=sr, hop_length=hop_length)

    return (inputnp, times)





def procMidiData(pmobj, times):

    outputnp = mid2outputnp(pmobj, times)

    return outputnp

def drawPianoRoll(pianoroll, title=''):
    # Create a `pypianoroll.Track` instance
    pianoroll = pianoroll.T
    track = Track(pianoroll=pianoroll, program=0, is_drum=False,
                  name=title)

    # Plot the piano-roll
    fig, ax = track.plot()
    plt.show()