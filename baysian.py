import numpy as np
import pandas as pd


NUMHYPOTHESIS = 25

#calculates posteriors from priors and likelihoods
def Posterior(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    return np.multiply(prior, likelihood)

#converts the string representation of the set to a set of ints
def str_to_set(str: str) -> set:
    split_str = set.split()
    int_list = [int(num.replace('_', '')) for num in split_str]
    int_set = set(int_list)

# TODO: need the hypothesis we'll be using first
# return a list of size NUMHYPOTHESIS where each index is the list of all targets that fit the hypothesis
def correct_sets(data: dict, set: str) -> list:
    set = str_to_set(set)
    

    ...

#returns a #sets by #hypotheses numpy array of likelihoods
def likelihood(data: dict) -> np.ndarray:
    correct = np.zeros((255,NUMHYPOTHESIS))
    total = np.zeros(255)
    for i, set in enumerate(data):
        hypotheses = correct_sets(data, set)
        for j, hypothesis in enumerate(hypotheses):
                correct_targets_yes = data[set][hypothesis, 1]
                mask = np.ones(100, dtype=bool)
                mask[hypothesis] = False
                correct_targets_no = data[set][mask, 0]
                correct[i][j] = np.sum(correct_targets_yes) + np.sum(correct_targets_no)
                total[i] = np.sum(data[set])

    return correct/total

#creates a dict of all the responses by set
def preprocess(data: pd.DataFrame) -> dict:
    results = {}
    for trial in data.itertuples():
        if trial.set not in results:
            results[trial.set] = np.zeros((100,2))
        results[trial.set][trial.target - 1][trial.rating] += 1
    return results


file = pd.read_csv('numbergame_data.csv')
data = preprocess(file)