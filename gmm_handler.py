import sklearn.mixture
import os


class GMMHandler:

    def __init__(self, data, n_mixtures, word):
        self.data = data
        self.n_mixtures = n_mixtures
        self.word = word
        self.num_obs = self.data.shape[0]

    def get_gmm(self, ob):
        g = sklearn.mixture.GaussianMixture(n_components=self.n_mixtures, max_iter=10000, tol=0.001)
        g.fit(ob.reshape(-1, 1))
        return g

    def get_observations(self):
        observations = []
        for state in range(self.data.shape[0]):
            observation = self.get_gmm(self.data[state])
            observations.append(observation)
        return observations

    def save_observations(self, observations):
        cwd = os.getcwd()
        filepath = cwd + f"/gmm_data/{self.word}.txt"
        file = open(filepath, "w")
        k = 0
        for gmm in observations:
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_
            means_str = ""
            weights_str = ""
            covariances_str = ""
            for i in range(weights.size):
                means_str += str(means[i][0]) + " "
                weights_str += str(weights[i]) + " "
                covariances_str += str(covariances[i][0][0]) + " "
            file.write(f"{k}_m={means_str}\n")
            file.write(f"{k}_w={weights_str}\n")
            file.write(f"{k}_c={covariances_str}\n")
            k += 1
        file.close()
