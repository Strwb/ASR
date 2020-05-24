from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

(rate, sig) = wav.read("one.wav")
mfcc_feat = mfcc(sig, rate, numcep=12, nfft=2400)
# plt.plot(mfcc_feat[0])
# plt.show()

# Boundaries for number of states per one HMM model
n_min = 5
n_max = 20

# Number of components in gaussian distribution mixture
n_gauss = 6

# Mutation operators
Pst = 0.6  # Probability of mutation of number of states
Pval = 0.1  # Probability of mutation of HMM values
variance = 1  # Mutation variance

# Reduction factors
k_st = 0.99
k_val = 0.99
k_v = 0.999

t = 0  # Mutation index

# Population of size 10, each population represents a single HMM model
# print(mfcc_feat.shape)
# print(mfcc_feat)
# print(np.histogram(mfcc_feat[0, :]))
hist1 = np.histogram(mfcc_feat)
print(hist1)
plt.hist(hist1)
plt.show()
