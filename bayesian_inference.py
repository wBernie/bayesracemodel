import numpy as np
import pandas as pd
from concepts import *

NUMHYPOTHESIS = 101
CONSTANT = 1e-6

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

def set_likelihood(s:set, hypotheses):
    likelihood = np.zeros((1, NUMHYPOTHESIS))
    for j, hypothesis in enumerate(hypotheses):
            h = set(hypothesis)
            common = s.intersection(h)
            likelihood[0][j] = len(common)/len(s)
    return likelihood

def sets_likelihood(sets: set, hypotheses) -> np.ndarray:
    '''
    given set of sets and a set of hypotheses for each set, returns the likelihood of each hypothesis for each set
    '''
    likelihood = np.zeros((255, NUMHYPOTHESIS))
    for i, s in enumerate(sets):
        likelihood[i] = set_likelihood(s, hypotheses[i])
    return likelihood

#returns a #sets by #hypotheses numpy array of likelihoods
def likelihood(data: dict, best_h, hypotheses) -> np.ndarray:
    correct = np.zeros((255))
    total = np.zeros(255)
    correct_s = np.zeros((255,100))
    total_s = np.zeros((255,100))
    for i, s in enumerate(data):
        hypothesis = hypotheses[i][best_h[i]]
        index = [i-1 for i in hypothesis]
        correct_targets_yes = data[s][index, 1]
        mask = np.ones(100, dtype=bool)
        mask[index] = False
        correct_targets_no = data[s][mask, 0]
        correct[i] = np.sum(correct_targets_yes) + np.sum(correct_targets_no)
        total[i] = np.sum(data[s])


        for j in range(100):
            correct_s[i][j] = j+1 in hypothesis
    return correct/total, correct_s

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

    likelihoods = sets_likelihood(sets_int, hypotheses)

    posteriors = calc_posterior(load_priors('cog260-project/data/priorsheet.csv'), likelihoods)
    return posteriors, load_priors('cog260-project/data/priorsheet.csv')

def info_gain():
    '''
    Returns a tuple of 606 x 15 x 31 x 101 array of posteriors and an array of 606 x 15 x 30 x 101 array of information gains

    the first index is participant #
    the 2nd index is set #
    3rd is number of targets seen, with information gain starting at 1 and posteriors starting at 0
    4th is concept #
    '''
    file = pd.read_csv('cog260-project/data/numbergame_data.csv')
    file = file.sort_values(by=['id', 'set'])

    data = preprocess(file)
    hypotheses = []
    sets_str = data.keys()
    sets_int = []
    for sets in sets_str:
        set_int = str_to_set(sets)
        sets_int.append(set_int)
        hypotheses.append(correct_sets(set_int))

    priors = load_priors('cog260-project/data/priorsheet.csv')
    dfhypotheses = pd.DataFrame(hypotheses)
    dfhypotheses['set'] = list(sets_str)    

    target_posteriors = np.zeros((len(file.id.unique()), 15, 31, 101))
    participant_i = -1
    s_i = -1
    t_i=1
    participant, s_orig, s = None, None, None
    for line in file.itertuples():
        if line.id != participant:
            participant = line.id
            participant_i += 1
            s_i = -1
        if line.set != s_orig or s_i == -1 or t_i == 31:
            s_orig = line.set
            s_i += 1
            t_i = 1
            s = line.set
            hypothesis = dfhypotheses[dfhypotheses['set'] == s].iloc[0].tolist()[1:]
            s = str_to_set(s)
            target_posteriors[participant_i][s_i][0] =  calc_posterior(priors, set_likelihood(s, hypothesis))
            
        
        s = s.union({line.target})
        target_posteriors[participant_i][s_i][t_i] = calc_posterior(priors, set_likelihood(s, hypothesis))
        t_i += 1
    
    target_posteriors += 1e-6
    info_gain = np.log(target_posteriors[:, :, 1:]/ target_posteriors[:, :, :-1])
    target_posteriors = np.delete(target_posteriors, 0, 2)
    return target_posteriors, info_gain

if __name__ == "__main__":
    file = pd.read_csv('cog260-project/data/numbergame_data.csv')
    file = file.sort_values(by=['id', 'set'])
    data = preprocess(file)
    hypotheses = []
    sets_str = data.keys()
    sets_int = []
    for sets in sets_str:
        set_int = str_to_set(sets)
        sets_int.append(set_int)
        hypotheses.append(correct_sets(set_int))

    likelihoods = sets_likelihood(sets_int, hypotheses)

    posteriors = calc_posterior(load_priors('cog260-project/data/priorsheet.csv'), likelihoods)

    import pprint
    pprint.pprint(posteriors)
    print(posteriors.shape)
    best_h = np.argmax(posteriors, axis=1)

    likelihoods, likelihoods_s = likelihood(data, best_h, hypotheses)
    print(likelihoods, likelihoods.shape)

    print(np.mean(likelihoods), np.var(likelihoods))

    answer = np.zeros((255,100))
    total = np.zeros((255, 100))
    set_str = list(sets_str)
    for l in file.itertuples():
        answer[list(sets_str).index(l.set), l.target -1] += 1*(l.rating)
        total[list(sets_str).index(l.set), l.target -1] += 1

    p_average = answer/total


    import matplotlib.pyplot as plt

    for s in range(255):
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot for model aggregate
        plt.bar(range(100), p_average[s], alpha=0.5, color='blue', label='Data')
        # Bar plot for data aggregate
        plt.bar(range(100), likelihoods_s[s], alpha=0.5, color='red', label='Bayes')
        
        
        
        # Set y-axis limits the same for all plots
        plt.ylim(0, 1.1)
        
        plt.title(f'Set {list(sets_str)[s]}')
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"cog260-project/figures_ig/bayesset{s}.png")
        plt.close()



    # info_gain()