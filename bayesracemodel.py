from pyClarion import Process, NumDict, Index, Event, numdict, bind, UpdateSite, KeyForm, Key, path, root, Agent
from pyClarion.knowledge import Family, Atoms, Atom
import math
from datetime import timedelta
from typing import *
from tqdm import tqdm

from bayesian_inference import b_inference, NUMHYPOTHESIS

class LogNormalRace(Process):

    class Params(Atoms):
        # a keyspace with two keys in it
        # define parameters as atoms. Idea: however you've configured your keyspaces, it shud definitely have these keys
        # this is like a required parameters class
        F: Atom #
        sd: Atom

    choice: NumDict
    sample: NumDict
    params: NumDict

# Family, Sort, and Terms are a hierarchy

    def __init__(self, 
                name:str, pfam:Family, posteriors, F: float = 1, sd: float=1.0) -> None:
        super().__init__(name)
        root = self.system.index
        root.a
        index = Index(root, f"a", (1,))
        populate_index(self)
        self.p = type(self).Params()
        pfam[name] =  self.p# assign the keyspace to this attr, under the processes own name
        # the params under p are now under p -> name -> params
        self.choice = numdict(index, {}, 0.0) #default 0 in the beginnning, 
        self.sample = numdict(index, {}, 0.0)
        self.params = numdict(Index(root(pfam), "p:?:?"), {f"p:{name}:F": F, f"p:{name}:sd": sd}, float("nan")) # coz i dont intend anything else to be there

    def resolve(self, event: Event) -> None:
        # a lognormal race to react to the presentation of stimulus
        if event.affects(self.params): # smol change here - since the inputs are going to be shared with outputs of CA model
            # there shudnt really be any other stimuli before i respond to the next stimulus
            if any(event.source == self.update for e in self.system.queue):
                raise RuntimeError("You made a scheduling mistake") # since i already intend to react to a stimulus that was already presetned 
            self.update()

    def update(self) -> None:
        sd = self.params[path(self.p.sd)] # get the path
        sample = self.sample.normalvariate(numdict(self.sample.i, {}, sd))
        choice = numdict(self.input.i, {sample.argmax(): 1.0}, 0) 
        rt = self.params[path(self.p.F)] * math.exp(-sample.valmax())

        self.system.schedule(self.update, UpdateSite(self.sample, sample.d), UpdateSite(self.choice, choice.d), dt=timedelta(milliseconds=rt))

"""
The above model is to be called after fitting the parameters. We fit the parameters by running a grid search. 
"""
from scipy.special import erfc
import math 
import numpy as np
import pandas as pd
def log_likelihood(rts: np.ndarray, concept_scores: np.ndarray, winning_concept_mask: np.ndarray, F_i: float, sd_i: float) -> pd.Series:
    # Calculate the mean and sigma values for all rows in the DataFrame
    mean_word = concept_scores + np.log(F_i)

    sigma_word = sd_i
    sigma_nword = sd_i
    
    # For correct responses (ACC = TRUE)
    #NOTE: numerous log simplications were made. 
    l_concept = (-np.log(rts * sigma_word * np.sqrt(2* np.pi)) - 
              ((np.log(rts) - mean_word[(mean_word["winning_concept_score"] * winning_concept_mask).sum(axis=-1)])**2) / (2 * sigma_word**2))
               
    for i in range(concept_scores.shape[1]):
        l_concept += np.log(1 - 0.5*erfc(-(np.log(rts) - mean_word[f"concept_{i}"]) / (sigma_nword * np.sqrt(2)))) * (1-winning_concept_mask[:, i:i+1])
    
    likelihood = np.where((-np.inf < likelihood) & (likelihood < np.inf), likelihood, np.nan)
    return np.nansum(likelihood)

#populate matrix with hypotheses:
def populate_index(model: LogNormalRace):
    global NUMHYPOTHESIS
    a_index = model.main.i
    for item in range(NUMHYPOTHESIS):
        getattr(a_index.root.a, str(item))

def populate_weights(weights_index, posteriors):
    weights = numdict(weights_index, {}, 0.0)

    for i in weights_index:
        key = i
        int_key = int(str(key).split(":")[-1])
        with weights.mutable() as d:
                d[key] = posteriors[int_key]

    return weights
#TODO: function to fit parameters:

#run simulation:   
def run_race_model_per_person(data_i, sets_int, posteriors, Fs, sds, p_idx=0):
    race = LogNormalRace("model")
    limit = timedelta(days=15)

    data = [] 
    choices = []
    time_sum = timedelta() # to get the true response time (else accumulated)
    #load up the first datapoint
    i = 0
    s =[int(m) for m in data_i.iloc[i]["set"].split("_ ")]
    sample = populate_weights(race.sample.i, posteriors[sets_int.index(s)])
    
    #there will be no populating input -- since considdering the input is done in creating the posterior
    params_data = numdict(race.params.i, {"p:model:F": Fs[p_idx], "p:model:sd": math.sqrt(sds[p_idx])}, 0)
    race.system.user_update(UpdateSite(race.sample, sample.d), UpdateSite(race.params, params_data.d))

    while (time_sum == timedelta()) or (race.system.queue and race.system.clock.time < limit):
        race.system.advance() #set the current 
        event = race.system.advance() # get the participant's choice
        data.append((s, event.time - time_sum)) # 1 if word is choice, else 0 
        choices.append(race.choice.valmax())
        time_sum = event.time
        #load the next data
        i+=1

        if len(data_i) > i:
            s =[int(m) for m in data_i.iloc[i]["set"].split("_ ")]
            sample = populate_weights(race.sample.i, posteriors[sets_int.index(s)])
            params_data = numdict(race.params.i, {"p:model:F": Fs[p_idx], "p:model:sd": sds[p_idx]}, 0)
        else:
            break
        race.system.user_update(UpdateSite(race.sample, sample.d), UpdateSite(race.params, params_data.d))
    return data, choices


def preprocess_targets_per_participant(sets_int, priors, hypotheses):
    df = pd.read_csv('cog260-project/data/numbergame_data.csv')
    participants = pd.unique(df["id"]) 
    df_dict = {"id":[], "set":[], "yes_targets":[], "no_targets":[], "avg_rt":[], "best_hyppothesis":[]}
    # 255 participants, each was shown 15 different sets, for each set 30 targets. 
    for participant in tqdm(participants): 
        p_df = df[df["id"] == participant]
        sets = pd.unique(p_df["set"])
        for s in sets:
            s_df = p_df[p_df["set"] == s]
            yes_set = pd.unique(s_df[s_df["rating"] == 1]["target"]).tolist()
            no_set = pd.unique(s_df[(s_df["rating"] == 0)]["target"]).tolist()

            assert list(yes_set) == yes_set 
            assert list(no_set) == no_set

            yes_set, no_set = set(yes_set), set(no_set)
            # assert not len(yes_set.intersection(no_set)), f"{yes_set} {no_set}" -- strangely enough there are a couple participants for whom theyr sample without replacement assumption does not hold,idk why
            avg_rt = s_df["rt"].mean()

            s_indx = sets_int.index(set([int(m) for m in s.split("_ ")]))
            assert s_indx != -1
            h_idx = best_hypothesis(priors, hypotheses[s_indx], yes_set, no_set)

            df_dict["id"].append(participant.item())
            df_dict["set"].append(s)
            df_dict["yes_targets"].append("_".join([str(m) for m in yes_set]))
            df_dict["no_targets"].append("_".join([str(m) for m in no_set]))
            df_dict["avg_rt"].append(avg_rt)
            df_dict["best_hyppothesis"].append(h_idx)

    df = pd.DataFrame.from_dict(df_dict)
    return df, participants

def best_hypothesis(priors, hypotheses, yes_set, no_set):
    max_h = None
    max_prior = 0
    max_idx = -1

    for i, h in enumerate(hypotheses):
        yes = len(yes_set.intersection(set(h)))
        no = len(no_set) - len(no_set.intersection(set(h)))
        measure = (yes+no)/len(no_set.union(yes_set))
        if not max_h or measure > max_h:
            max_h = measure
            max_idx = i
            max_prior = priors[0, i]
        elif measure == max_h and priors[0, i] >  max_prior:
            max_h = measure
            max_idx = i
            max_prior = priors[0, i]
    return max_idx

def setup():
    posteriors, priors, hypotheses, sets_int = b_inference()
    new_df, participants = preprocess_targets_per_participant(sets_int, priors, hypotheses)

    #TODO: fitting parameters sd and F

    for i, p in enumerate(participants):
        run_race_model_per_person(new_df[new_df["id"] == p.item()], sets_int, posteriors, Fs, sds, p_idx=i)



