from pyClarion import Process, NumDict, Index, Event, numdict, bind, UpdateSite, KeyForm, Key, path, root, Agent
from pyClarion.knowledge import Family, Atoms, Atom
import math
from datetime import timedelta
from typing import *
from tqdm import tqdm

from bayesian_inference import b_inference, NUMHYPOTHESIS
from fitting_params import gradient_descent

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
import math 
import numpy as np
import pandas as pd

#populate matrix with hypotheses:
def populate_index(model: LogNormalRace):
    global NUMHYPOTHESIS
    a_index = model.main.i
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
def run_race_model_per_person(data_i, sets_int, posteriors, Fs, sds, ths, p_idx=0):
    race = LogNormalRace("model")
    limit = timedelta(days=15)

    data = [] 
    choices = []
    time_sum = timedelta() # to get the true response time (else accumulated)
    #load up the first datapoint
    i = 0
    s =[int(m) for m in data_i.iloc[i]["set"].split("_ ")]
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
    #TODO: just like bernie does. 
    pass

def setup():
    posteriors, sets_int = b_inference() # posterios are of size 606x15x30x101
    df = pd.read_csv("cog260-project/data/number_game_data.csv")
    #drop unnecessary cols:
    df.drop(labels=["age", "firstlang", "gender", "education", "zipcode"])
    participants = pd.unique(df["id"]).tolist()

    # fit_parameters(df, participants, posteriors, sets_int)
    #participant one hot matrix
    participant_in = torch.eye((len(participants, participants)))

    targets = get_target(df, participants)
    gradient_descent(participant_in, targets, posteriors)

    # run simulation per participant
    corrects, data = [], []

    for i, p in enumerate(participants):
        d, choices = run_race_model_per_person(df[df["id"] == p], sets_int, posteriors, Fs, sds, i)
        data += d
        corrects += (np.array(choices) == df[df["id"] == p]["rating"]).tolist()
    
    ca_rate = 100 * sum(corrects)/len(corrects)



