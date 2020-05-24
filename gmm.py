from random import uniform
from math import sqrt, log, exp, pi


class Gaussian:
    """Gaussian probability density function"""

    def __init__(self, mu, sigma):
        # Mean and standard deviation of the distribution
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        """Probability of a data point given the current parameters"""
        u = (datum-self.mu) / abs(self.sigma)
        y = (1 / sqrt(2 * pi) * abs(self.sigma)) * exp((-u * u) / 2)
        return y


class GMM:
    def __init__(self, data, mu_min=None, mu_max=None, sigma_min=.1, sigma_max=1, mix=.5):
        if mu_min is None:
            mu_min = min(data)
        if mu_max is None:
            mu_max = max(data)
        self.data = data
        self.one = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.mix = mix

    def EStep(self):
        """Perform an E(stimation) step, freshening up self.loglike in the process"""
        # Compute weights
        self.loglike = 0.
        for datum in self.data:
            # Unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)
            # Compute denominator
            den = wp1 + wp2
            # Normalize
            wp1 /= den
            wp2 /= den
            # Add into loglike
            self.loglike += log(wp1 + wp2)
            # Yield weight tuple
            yield (wp1, wp2)  # Instead of yield we can call Mstep on these weights

    def MStep(self, weights):
        """Perform a M(aximization) step"""
        # Compute denominators
        (left, right) = zip(*weights) # TODO O co cho z tymi weightami
        one_den = sum(left)
        two_den = sum(right)
        # Compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, self.data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(right, self.data))
        # Compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, self.data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(right, self.data)) / two_den)
        # Compute new mix
        self.mix = one_den / len(self.data)

    def iterate(self, N=1):
        """Perform N iterations, then compute log-likelihood"""
        pass # TODO

    def pdf(self, x):
        return (self.mix)*self.one.pdf(x) + (1-self.mix)*self.two.pdf(x)
