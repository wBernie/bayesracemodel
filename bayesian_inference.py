import numpy as np
import pandas as pd
from concepts import *

NUMHYPOTHESIS = 101

def load_priors(csv_file: str) -> np.ndarray:
    """
    Docstring goes here:

    """
    df = pd.read_csv(csv_file)
    #remove the comments column 
    df.drop(columns=['comments'], inplace=True)
    #remvove the rows where the value of "used" is "no"
    df = df[df['used'] != 'no']
    #drop the "used" column now
    df.drop(columns=["used"],inplace=True)

    #normalize:
    df = df["count"].to_numpy()[np.newaxis, :]
    df = df/df.sum()

    return df

#calculates posteriors from priors and likelihoods
def calc_posterior(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    return np.multiply(prior, likelihood)

#converts the string representation of the set to a set of ints
def str_to_set(str: str) -> set:
    split_str = str.split()
    int_list = [int(num.replace('_', '')) for num in split_str]
    int_set = set(int_list)
    return int_set

# return a list of size NUMHYPOTHESIS where each index is the list of all targets that fit the hypothesis
#(sorry)
def correct_sets(set: str) -> list:
    lists = []
    lists.append(even())
    lists.append(odd())
    lists.append(btw(set))
    lists.append(primes())
    lists.append(nonprimes())
    lists.append(same(set))
    lists.append(twodigit())
    lists.append(onedigit())
    lists.append(odd_sum())
    lists.append(not_multiples(3))
    mult = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 25, 29, 33, 100]
    for i in mult:
        lists.append(multiples(i))
    cont = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in cont:
        lists.append(contains(i))
    lists.append(contains_even())
    betweens = [[32, 100], [5, 100], [79, 100], [1, 81], [26, 100],[1, 49], [1, 9]]
    for i in betweens:
        lists.append(btw(start = i[0], end = i[1]))
    lists.append(math(2, 3))
    lists.append(math(2, 1))
    lists.append([93])
    betweens = [[1, 29],[1,64], [41, 100],[1, 99], [99, 100], [92, 100], [1, 3], [1, 82]]
    for i in betweens:
        lists.append(btw(start = i[0], end = i[1]))
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in digits:
        lists.append(starts(i))
    lists.append(starts_even())
    lists.append(starts_odd())
    for i in cont:
        lists.append(ends(i))
    lists.append(ends_even())
    lists.append(ends_odd())
    lists.append(start_end())
    lists.append(sum_digit_eq(10))
    lists.append(sum_digit_eq(9))
    isList = [18, 84, 48, 43, 91, 31]
    for i in isList:
        lists.append([i])
    lists.append(math(3, 1))
    betweens = [[51, 100], [64, 100], [82, 100], [1, 42], [52, 100], [1, 82], [1, 49], [1, 44], [1, 69], [71, 100]]
    for i in betweens:
        lists.append(btw(start = i[0], end = i[1]))
    return lists

def set_likelihood(sets: set, hypotheses) -> np.ndarray:
    likelihood = np.zeros((255, NUMHYPOTHESIS))
    for i, s in enumerate(sets):
        for j, hypothesis in enumerate(hypotheses[i]):
            h = set(hypothesis)
            common = s.intersection(h)
            likelihood[i][j] = len(common)/len(s)
    return likelihood

#returns a #sets by #hypotheses numpy array of likelihoods
def likelihood(data: dict, best_h, hypotheses) -> np.ndarray:
    correct = np.zeros((255))
    total = np.zeros(255)
    for i, set in enumerate(data):
        hypothesis = hypotheses[i][best_h[i]]
        index = [i-1 for i in hypothesis]
        correct_targets_yes = data[set][index, 1]
        mask = np.ones(100, dtype=bool)
        mask[index] = False
        correct_targets_no = data[set][mask, 0]
        correct[i] = np.sum(correct_targets_yes) + np.sum(correct_targets_no)
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


def b_inference():
    file = pd.read_csv('cog260-project/data/numbergame_data.csv')
    data = preprocess(file)
    hypotheses = []
    sets_str = data.keys()
    sets_int = []
    for sets in sets_str:
        set_int = str_to_set(sets)
        sets_int.append(set_int)
        hypotheses.append(correct_sets(set_int))

    likelihoods = set_likelihood(sets_int, hypotheses)

    posteriors = calc_posterior(load_priors('cog260-project/data/priorsheet.csv'), likelihoods)
    return posteriors, load_priors('cog260-project/data/priorsheet.csv'), hypotheses, sets_int

if __name__ == "__main__":
    file = pd.read_csv('cog260-project/data/numbergame_data.csv')
    data = preprocess(file)
    hypotheses = []
    sets_str = data.keys()
    sets_int = []
    for sets in sets_str:
        set_int = str_to_set(sets)
        sets_int.append(set_int)
        hypotheses.append(correct_sets(set_int))

    likelihoods = set_likelihood(sets_int, hypotheses)

    posteriors = calc_posterior(load_priors('cog260-project/data/priorsheet.csv'), likelihoods)

    import pprint
    pprint.pprint(posteriors)
    print(posteriors.shape)
    best_h = np.argmax(posteriors, axis=1)

    likelihoods = likelihood(data, best_h, hypotheses)
    print(likelihoods, likelihoods.shape)
    