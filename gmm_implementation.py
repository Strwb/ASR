import numpy as np
from math import sqrt, pi
from random import uniform


class Gaussian:

    def __init__(self, mu, var, weight=1):  # var = variance of the gaussian
        self.mu = mu
        self.var = var
        self.weight = weight
        self.b = []

    def pdf(self, x):
        power = -((x - self.mu)**2 / (2 * self.var))
        base = 1 / (sqrt(2*pi*self.var))
        density = base*np.exp(power)
        return density

    def clear_b(self):
        """Clear distribution probabilities for each run of the algorithm"""
        self.b = []


class GMM:

    def __init__(self, data, n):
        self.data = data
        self.gaussians = []
        mu_l = min(self.data)  # Lower bound of randomly generated mean
        mu_u = max(self.data)  # Upper bound of randomly generated mean
        v_l = 0.1  # Lower bound of randomly generated variance
        v_u = 1  # Upper bound of randomly generated variance
        for i in range(n):  # For each gaussian that we want to have
            rmu = uniform(mu_l, mu_u)  # Random initialization of mean
            rvar = uniform(v_l, v_u)  # Random initialization of variance
            self.gaussians[i] = Gaussian(rmu, rvar, 1/n)

    def calculate_b(self, x):
        """For each x in data set we calculate probability of belonging
        to each distribution"""
        denominator = 0
        numerators = []
        for g in self.gaussians:
            prob = g.pdf(x)  # Calcu
            weight = g.weight
            numerators.append(prob * weight)
            denominator += prob * weight
        for k in range(0, len(self.gaussians)):
            bk = numerators[k] / denominator
            self.gaussians[k].b.append(bk)

    def training(self):
        for g in self.gaussians:  # Clear distribution probabilities
            g.clear_b()
        for x in self.data:  # For each datapoint perform E and M step
            self.Estep(x)
            self.Mstep(x)

    def Estep(self, x):  # In this step we calculate B
        self.calculate_b(x)

    def Mstep(self, x):
        """At every datapoint recalculate gaussian parameters using
        distribution probabilities calculated earlier"""
        for k in range(len(self.gaussians)):
            g = self.gaussians[k]
            sum_b = sum(g.b)
            mean_nom = 0
            var_nom = 0
            for i in range(len(g.b)):
                mean_nom += g.b(i) * x
                var_nom += g.b(i) * (x - g.mu)**2
            g.mu = mean_nom / sum_b
            g.var = var_nom / sum_b
            g.weight = sum_b/len(self.data)
