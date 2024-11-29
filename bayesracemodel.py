from pyClarion import Process, NumDict, Index, Event, numdict, bind, UpdateSite, KeyForm, Key, path, root, Agent
from pyClarion.knowledge import Family, Atoms, Atom
import math
from datetime import timedelta
from typing import *
from tqdm import tqdm

from bayesian_inference import info_gain, NUMHYPOTHESIS
from fitting_params import gradient_descent, log_likelihood

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
                name:str, pfam:Family, F: float = 1, sd: float=1.0) -> None:
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

        self.system.schedule(self.update, UpdateSite(self.sample, sample.d), UpdateSite(self.choice, choice.d), dt=timedelta(seconds=rt)) # bcz i divided by 1000

"""
The above model is to be called after fitting the parameters. We fit the parameters by running a grid search. 
"""
import math 
import numpy as np
import pandas as pd

#populate matrix with hypotheses:
def populate_index(model: LogNormalRace):
    global NUMHYPOTHESIS
    a_index = model.sample.i
    for item in range(NUMHYPOTHESIS):
        getattr(a_index.root.a, str(item))
    a_index.root.a.th # decision threshhold

def populate_weights(weights_index, posteriors, th):
    weights = numdict(weights_index, {}, 0.0)

    with weights.mutable() as d:
        for i in weights_index:
            key = i
            if "th" in str(key):
                d[key] = th
            else:
                int_key = int(str(key).split(":")[-1])
                d[key] = posteriors[int_key]
    return weights

#run simulation:   
def run_race_model_per_person(data_i, posteriors, Fs, sds, ths, p_idx=0):
    p = Family()
    with Agent("agento") as agent:
        agent.system.root.p = p 
        race = LogNormalRace("model", p)
    limit = timedelta(days=15)

    data = [] 
    choices = []
    time_sum = timedelta() # to get the true response time (else accumulated)
    #load up the first datapoint
    i = 0
    s = data_i.iloc[i]["set"]
    sets_int = pd.unique(data_i["set"])
    sample = populate_weights(race.sample.i, posteriors[p_idx, sets_int.index(s), i], ths[p_idx])
    
    #there will be no populating input -- since considdering the input is done in creating the posterior
    params_data = numdict(race.params.i, {"p:model:F": Fs[p_idx], "p:model:sd": math.sqrt(sds[p_idx])}, 0)
    race.system.user_update(UpdateSite(race.sample, sample.d), UpdateSite(race.params, params_data.d))

    while (time_sum == timedelta()) or (race.system.queue and race.system.clock.time < limit):
        race.system.advance() #set the current 
        event = race.system.advance() # get the participant's choice
        data.append((s, event.time - time_sum, race.choice.argmax)) # 1 if word is choice, else 0 
        choices.append("th" not in str(race.choice.argmax())) # 0 is th, else 1
        time_sum = event.time
        #load the next data
        i+=1

        if len(data_i) > i:
            s = [int(m) for m in data_i.iloc[i]["set"].split("_ ")]
            sample = populate_weights(race.sample.i, posteriors[p_idx, sets_int.index(s), i], ths[p_idx])
            params_data = numdict(race.params.i, {"p:model:F": Fs[p_idx], "p:model:sd": sds[p_idx]}, 0)
        else:
            break
        race.system.user_update(UpdateSite(race.sample, sample.d), UpdateSite(race.params, params_data.d))
    return data, choices
import torch
def get_target(df, participants: List[int]) -> torch.tensor:
    targets = np.zeros((len(participants), 15, 30, 2))
    flag_60_, flag_60 = 0, 0
    for i, p in tqdm(enumerate(participants), total=len(participants)):
        p_df = df[df["id"] == p]
        sets_int = pd.unique(p_df["set"])
        for j, s in enumerate(sets_int):
            s_df = p_df[p_df["set"] == s]
            k=0 
            for _, l in s_df.iterrows():
                assert type(l["rt"]) is float, f"{i} {j} {k}"
                assert l["rt"]/1000 is not np.nan, f"{i} {j} {k} {l["rt"]}"
                targets[i, j + flag_60_, k - 30*flag_60, 0] = l["rt"]/1000
                targets[i, j + flag_60_, k-30*flag_60, 1] = l["rating"] == "yes"
                k+=1
                if k == 30 and len(s_df) == 60:
                    flag_60 = 1
                    flag_60_ = 1
            flag_60=0
        flag_60_ = 0
    targets = np.where(np.isnan(targets), 400, targets)
    targets = np.where(targets == 0, 400, targets)
    return targets

def estimate_F(scores, targets):
    p_df_rt = targets.reshape(606, -1)
    mean_logF = np.nanmean(np.log(p_df_rt), axis=-1) + np.nanmean(scores.max(axis=-1).reshape(606, -1))
    return np.exp(mean_logF)

def estimate_th(targets, F):
    p_df_rt = targets.reshape(606, -1)
    mean_th = np.log(F) - np.nanmean(np.log(p_df_rt), axis=-1)
    return mean_th

def estimate_sd(targets, F, th):
    p_df_rt = targets.reshape(606, -1)
    st = np.nanstd(p_df_rt, axis=-1)/(F**2)
    st = np.sqrt(np.log(1+(st**2)/(th**2))) # https://math.stackexchange.com/questions/4658759/lognormal-distribution-mean-and-variance-of-logarithm-of-distribution
    return st

def estimate_parameters(scores, targets):
    print("Estimating parameters!")
    rts, choices = targets[:, :, :, 0],targets[:, :, :, 1] 
    min_F, min_th, min_sd = estimate_F(scores, rts), estimate_th(rts, estimate_F(scores, rts)), estimate_sd(rts, estimate_F(scores, rts), estimate_th(rts, estimate_F(scores, rts)))
    F, TH, SD = min_F, min_th, min_sd
    max_lk = -np.inf
    for f in tqdm(np.arange(-0.3, 0.3, 0.1)):
        f = np.where((F+f) > 0, F+f, F)
        for sd in np.arange(-1, 1, 0.3):
            sd = SD + sd
            for th in np.arange(-5, 5, 2.5):
                th = th+TH
                lk = log_likelihood(torch.from_numpy(rts), torch.from_numpy(scores), torch.from_numpy(choices), torch.from_numpy(f), torch.from_numpy(sd), torch.from_numpy(th))
                if lk > max_lk:
                    max_lk = lk
                    min_F, min_th, min_sd =  f, th, sd
    return min_F, min_th, min_sd

import matplotlib.pyplot as plt
import os
def plot_distribution(df, choices, participants):
    #255 sets, 100 numbers 
    data_aggregate = np.zeros((255, 100))
    model_aggregate = np.zeros((255, 100))

    data_aggregate_counts = np.zeros((255, 100))

    set_ints = pd.unique(df["set"])

    for i, p in tqdm(enumerate(participants), total=len(participants)):
        p_df = df[df["id"] == p]
        sets_int = pd.unique(p_df["set"])
        for j, s in enumerate(sets_int):
            s_df = p_df[p_df["set"] == s]
            k=0 
            for _, l in s_df.iterrows():
                data_aggregate[set_ints.index(l["set"]), l["target"] - 1] += 1*(l["rating"])
                model_aggregate[set_ints.index(l["set"]), l["target"] - 1] += choices[i][(j + flag_60_)*30 + k - 30*flag_60]

                data_aggregate_counts[set_ints.index(l["set"]), l["target"] - 1] += 1
                k+=1
                if k == 30 and len(s_df) == 60:
                    flag_60 = 1
                    flag_60_ = 1
            flag_60=0
        flag_60_ = 0

    data_aggregate = data_aggregate / data_aggregate_counts
    model_aggregate = model_aggregate/data_aggregate_counts

   # Plotting separate figures for each set
    for i in range(255):
        plt.figure(figsize=(10, 6))
        
        # Bar plot for data aggregate
        plt.bar(range(100), data_aggregate[i], alpha=0.5, color='blue', label='Data')
        # Bar plot for model aggregate
        plt.bar(range(100), model_aggregate[i], alpha=0.5, color='red', label='Model')
        
        # Set y-axis limits the same for all plots
        plt.ylim(0, 1)
        
        plt.title(f'Set {i+1}')
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"cog260-project/figures/set{i}.png")


def setup():
    # target_posteriors, inf_gain = info_gain() # posterios are of size 606x15x30x101
    target_posteriors = np.load("cog260-project/data/target_posts.npy")
    inf_gain = np.load("cog260-project/data/inf_gain.npy")
    target_posteriors -= 1e-6
    target_posteriors = np.log(target_posteriors/(1-target_posteriors)) # logit
    target_posteriors = target_posteriors[:, :, 1:, :]


    df = pd.read_csv("cog260-project/data/numbergame_data.csv")
    #drop unnecessary cols:
    df.drop(labels=["age", "firstlang", "gender", "education", "zipcode"], axis=1, inplace=True)
    df.sort_values(by=['id', 'set'], inplace=True)

    participants = pd.unique(df["id"]).tolist()


    targets = get_target(df, participants)
    # s_table = gradient_descent(participant_in, torch.from_numpy(targets), torch.from_numpy(inf_gain))
    # torch.save(s_table, "s_table_proj.pt")
    # Fs, ths, sds = s_table[:, -1].tolist(), s_table[:, -2].tolist(), s_table[:, -3].tolist()
    scores = target_posteriors
    Fs, ths, sds = estimate_parameters(scores, targets)

    # run simulation per participant
    corrects, chcs, data = [], [], []
    print("Fitting complete, running race model")
    for i, p in tqdm(enumerate(participants)):
        d, choices = run_race_model_per_person(df[df["id"] == p], scores, Fs, sds, ths, i)
        data.append(d)
        chcs.append([choices])
        corrects += (np.array(choices) == df[df["id"] == p]["rating"]).tolist()
    
    ca_rate = 100 * sum(corrects)/len(corrects)
    print(f"Correctness rate: {ca_rate}")

    plot_distribution(df, chcs, participants)

if __name__ == "__main__":
    setup()