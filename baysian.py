import numpy as np
import pandas as pd
from bayesracemodel import load_priors

NUMHYPOTHESIS = 101

#calculates posteriors from priors and likelihoods
def Posterior(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    return np.multiply(prior, likelihood)

#converts the string representation of the set to a set of ints
def str_to_set(str: str) -> set:
    split_str = str.split()
    int_list = [int(num.replace('_', '')) for num in split_str]
    int_set = set(int_list)
    return int_set

def even() -> list:
    return [2 * i for i in range(1, 51)]
    # if all(item % 2 == 0 for item in set):
    #     return [2 * i for i in range(1, 51)]
    # return []

def odd() -> list:
    return [2 * i + 1 for i in range(0, 50)]
    # if all(item % 2 == 1 for item in set):
    #     return [2 * i + 1 for i in range(0, 50)]
    # return []


def btw(set = {}, start = None, end = None) -> list:
    if start and end:
        maxi = end
        mini = start
    else:
        maxi = max(set)
        mini = min(set)
    return list(range(mini, maxi + 1))


def primes() -> list:
    prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    return list(prime)
    # if set <= prime:
    #     return list(prime)
    # return []

def nonprimes() -> list:
    return [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100]
    non_prime_set = {1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100}
    if set <= non_prime_set:
        return list(non_prime_set)
    return []

def same(set: set) -> list:
    return list(set)

def twodigit() -> list:
    return list(range(10, 100))
    # two_digit_numbers = set(range(10, 100))
    # if set <= two_digit_numbers:
    #     return list(two_digit_numbers)
    # return []

def onedigit() -> list:
    return list(range(1, 10))
    # one_digit_numbers = set(range(1, 10))
    # if set <= one_digit_numbers:
    #     return list(one_digit_numbers)
    # return []

def multiples(num: int) -> list:
    return list(range(num, 101, num))
    # possible = set(range(1, 101, num))
    # if set <= possible:
    #     return list(possible)
    # return []

def not_multiples(n: int) -> list:
    return [num for num in range(1,101) if num % n != 0]

def contains(digit: int):
    digit_str = str(digit)
    result_set = [num for num in range(1, 101) if digit_str in str(num)]
    return result_set

def contains_even_digit(num):
    even_digits = {'2', '4', '6', '8'} 
    return any(digit in even_digits for digit in str(num))


def contains_even():
    result_set = [num for num in range(1, 101) if contains_even_digit(num)]
    return result_set

def starts(digit: int):
    digit_str = str(digit)
    results_set = [num for num in range(1, 101) if str(num).startswith(digit_str)]
    return results_set

def starts_even():
    return [num for num in range(1, 101) if int(str(num)[0]) % 2 == 0]

def starts_odd():
    return [num for num in range(1, 101) if int(str(num)[0]) % 2 != 0]

def ends(digit: int):
    return [num for num in range(1, 101) if num % 10 == digit]

def ends_even():
    return [num for num in range(1, 101) if int(str(num)[-1]) % 2 == 0]

def ends_odd():
    return [num for num in range(1, 101) if int(str(num)[-1]) % 2 != 0]

#might want to add this to multiples
def start_end():
    output = list(range(1,10))
    output.append(num for num in range(11, 101, 11))
    return output

def sum_digit(num: int):
    return sum(int(digit) for digit in str(num))

def odd_sum():
    return [num for num in range(1, 101) if sum_digit(num) % 2 == 1]

def math(exponent: int, multiplier: int):
    return [num for num in range(1, 101) if (num/multiplier ** 1/exponent).is_integer()]

def sum_digit_eq(n: int):
    return [num for num in range(1,101) if sum_digit(num) % n == 0]



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
def likelihood(data: dict, best_h) -> np.ndarray:
    correct = np.zeros((255))
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


file = pd.read_csv('cog260-project/numbergame_data.csv')
data = preprocess(file)
hypotheses = []
sets_str = data.keys()
sets_int = []
for sets in sets_str:
    set_int = str_to_set(sets)
    sets_int.append(set_int)
    hypotheses.append(correct_sets(set_int))

likelihoods = set_likelihood(sets_int, hypotheses)

posteriors = Posterior(load_priors('cog260-project/260concepts - priorsheet.csv'), likelihoods)

import pprint
pprint.pprint(posteriors)
best_h = np.argmax(posteriors, axis=1)

