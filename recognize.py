import os
from import_processing import FileHandler
from gmm_handler import GMMHandler


class Recognizer:

    def __init__(self, num_obs):
        self.num_obs = num_obs

    def read_observations(self, filename):
        cwd = os.getcwd()
        filepath = cwd + f"/gmm_data/{filename}"
        file = open(filepath, "r")
        gaussians = []
        for ng in range(self.num_obs):

            linem = file.readline()
            linew = file.readline()
            linec = file.readline()

            means_str = linem[4:].split()
            means = [float(i) for i in means_str]

            weights_str = linew[4:].split()
            weights = [float(i) for i in weights_str]

            covariances_str = linec[4:].split()
            covariances = [float(i) for i in covariances_str]

            gaussians.append([means, weights, covariances])
        file.close()
        return gaussians

    def get_test_data(self, word):
        dirname = word + "_t"
        fh = FileHandler(dirname, 3)
        test_features = fh.extract()
        gh = GMMHandler(test_features, 3, word)
        test_data = gh.get_observations()
        gaussians = []

        for gmm in test_data:
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_

            m_vector = []
            w_vector = []
            c_vector = []

            for i in range(weights.size):
                m_vector.append(means[i][0])
                w_vector.append(weights[i])
                c_vector.append(covariances[i][0][0])
            gaussians.append([m_vector, w_vector, c_vector])
        return gaussians

    def compare_data(self, test_data, trained_data):
        difference = 0
        for i in range(self.num_obs):
            row_trained = trained_data[i]
            row_testing = test_data[i]
            for k in range(3):
                parameters_trained = sorted(row_trained[k])
                parameters_test = sorted(row_testing[k])
                for j in range(3):
                    difference += abs(parameters_trained[j]-parameters_test[j])
        return difference

    def recognize_word(self, word):
        test_data = self.get_test_data(word)
        cwd = os.getcwd()
        word_path = cwd + f"/gmm_data"
        wordnames = os.listdir(word_path)
        results_dictionary = dict()
        for name in wordnames:
            trained_data = self.read_observations(name)
            results_dictionary[name[:-4]] = self.compare_data(test_data, trained_data)
        result_string = ""
        smallest_key = ""
        smallest_data = 1000
        for key in results_dictionary:
            current_result = results_dictionary[key]
            if current_result < smallest_data:
                smallest_key = key
                smallest_data = current_result
            result_string += f"{key} -> {current_result} \n"
        print(result_string)
        print(f"Word predicted: {smallest_key}, with difference of {smallest_data}")

    def training(self, words):
        n_mixtures = 3
        for word in words:
            fh = FileHandler(word, self.num_obs)
            extracted_data = fh.extract()

            gh = GMMHandler(extracted_data, n_mixtures, word)
            gmm = gh.get_observations()
            gh.save_observations(gmm)
