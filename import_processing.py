import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank


class FileHandler:

    def __init__(self, word, parts):
        self.parts = parts
        cwd = os.getcwd()
        self.word_path = cwd + f"/words/{word}"
        self.wordnames = os.listdir(self.word_path)

    def word_features(self, word):
        path = self.word_path + f"/{word}"
        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate, nfft=2400)
        mfcc_fbank = logfbank(sig, rate, nfft=2400)

        vectors_num = np.size(mfcc_fbank, 0)
        feats = np.append(mfcc_feat[0], mfcc_fbank[0])
        for i in range(1, vectors_num):
            f_row = np.concatenate((mfcc_feat[i], mfcc_fbank[i]))
            feats = np.vstack((feats, f_row))

        step = vectors_num//self.parts
        last = 0
        features = np.zeros((self.parts, (step*39)+38))
        for i in range(self.parts):

            word_part = np.concatenate(feats[last:last+step])
            features[i, 0:word_part.size] = word_part
            last += step
        return features

    def extract(self):
        first_word = self.word_features(self.wordnames[0])
        for i in range(1, len(self.wordnames)):
            word = self.word_features(self.wordnames[i])
            first_word = np.concatenate((first_word, word), axis=1)
        return first_word
