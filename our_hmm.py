import random
import numpy as np


class State():
    def __init__(self, num_comp, o_size):
        self.mix_coeff = np.zeros(num_comp)
        self.mean_vec = np.zeros((M, o_size))
        self.covariance = np.zeros((M, o_size))

    def gen_mix_coeff(self):
        for i in range(self.mix_coeff.size):
            self.mix_coeff[i] = random_uniform(0.0001, 1)

    def gen_mean_vec =


Class Hmm():
    def __init__(self, o_size):
        self.num_states = random.randrange(5, 20)
        self.num_mix = 6
        self.tr = np.zeros((num_states, 2))
        self.opv = self.g_ob_prob(num_mix, o_size)

    def g_ob_prob(self, num_mix, o_size):
        observations = []
        for i in range(num_mix)
        observations[i] = State(num_mix, o_size)


