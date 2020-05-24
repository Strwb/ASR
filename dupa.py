from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

(rate, sig) = wav.read("one.wav")
mfcc_feat = mfcc(sig, rate, nfft=2400)
mfcc_fbank = logfbank(sig, rate, nfft=2400)

# plt.hist(mfcc_feat[0:14])
print(np.mean(mfcc_feat[0:14]))
# plt.show()

# print(mfcc_feat[0:14])
