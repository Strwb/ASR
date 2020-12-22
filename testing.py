import gmm_implementation as gmm
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import sklearn.mixture
import scipy.stats as stats


def pdf(data, mu, sd):
    var = sd**2
    output = []
    for x in data:
        power = -((x - mu)**2 / (2 * var))
        base = 1 / (sqrt(2*np.pi*var))
        output.append(base*np.exp(power))
    return output



(rate, sig) = wav.read("4A_endpt.wav")
mfcc_feat = mfcc(sig, rate, nfft=2400)
mfcc_fbank = logfbank(sig, rate, nfft=2400)

vector_num = np.size(mfcc_fbank, 0)

features = np.append(mfcc_feat[0], mfcc_fbank[0])

for i in range(1, vector_num):
    f_row = np.concatenate((mfcc_feat[i], mfcc_fbank[i]))
    features = np.vstack((features, f_row))

xd = np.concatenate(features[20:40])
xd = xd.reshape(-1, 1)

g = sklearn.mixture.GaussianMixture(n_components=3)
g.fit(xd)

weights = g.weights_
means = g.means_
covars = g.covariances_

x_axis = np.sort(xd)

plt.hist(xd, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)

for i in range(len(weights)):
    plt.plot(x_axis, pdf(x_axis, means[i], sqrt(covars[i])), ".")

plt.show()





