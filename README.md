# Polyphonic Piano Music Transcription
The final project of 2019 NCTU Deep Learning and Practice (Spring).

**Group Member:** Jennifer Wang, Hike Chen(ds282547)

# Introduction
Automatic Music Transcription (AMT) automates the process of converting an acoustic musical signal into some form of musical notation.
The AMT problem can be divided into several subtasks: multi-pitch detection, note onset/offset detection, quantization … etc.

We implemented a Hybrid model proposed by [1] to convert piano audio into musical notation in midi format. \
More details in [slides](https://github.com/ds282547/DLFinal/blob/main/slide/slide.pdf).

# Transcription Result Video

### Song 1 - Album audio:
1 Billion Lightyear of Distance \
[Source](https://www.youtube.com/watch?v=xMvdcnKzSa4) \
[Result](https://www.youtube.com/watch?v=XNAYiqEy_iM)
### Song 2 - Youtube video:

ちょっとたのしい「千本桜（Senbonzakura）」 を弾いてみた【ピアノ】\
[Source](https://www.youtube.com/watch?v=FnzoMzA9Dpg) \
[Result](https://www.youtube.com/watch?v=FnzoMzA9Dpg)
### Song 3 - Video recorded by mobile:
[Source](https://www.youtube.com/watch?v=C3I3JfmwBJk) 
[Result](https://www.youtube.com/watch?v=H8a9tebsHLo)

# Reference
[1] Sigtia, Siddharth, Emmanouil Benetos, and Simon Dixon. "An end-to-end neural network for polyphonic piano music transcription." IEEE/ACM Transactions on Audio, Speech, and Language Processing 24.5 (2016): 927-939.
