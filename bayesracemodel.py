from pyClarion import Process, NumDict, Index, Event, numdict, bind, UpdateSite, KeyForm, Key, path, root, Agent
from pyClarion.knowledge import Family, Atoms, Atom
import math
from datetime import timedelta
from typing import *
from tqdm import tqdm

from bayesian_inference import info_gain, NUMHYPOTHESIS
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
def run_race_model_per_person(data_i, posteriors, Fs, sds, ths, p_idx=0):
def run_race_model_per_person(data_i, posteriors, Fs, sds, ths, p_idx=0):
    race = LogNormalRace("model")
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

def setup():
    # target_posteriors, inf_gain = info_gain() # posterios are of size 606x15x30x101
    target_posteriors = np.load("cog260-project/data/target_posts.npy")
    inf_gain = np.load("cog260-project/data/inf_gain.npy")
    target_posteriors -= 1e-6
    target_posteriors = np.log(target_posteriors/(1-target_posteriors)) # logit
    target_posteriors = target_posteriors[:, :, 1:, :]


    df = pd.read_csv("cog260-project/data/numbergame_data.csv")
    #drop unnecessary cols:
    df.drop(labels=["age", "firstlang", "gender", "education", "zipcode"], axis=1)
    df.sort_values(by=['id', 'set'], inplace=True)

    participants = pd.unique(df["id"]).tolist()

    # fit_parameters(df, participants, posteriors, sets_int)
    #participant one hot matrix
    participant_in = torch.eye(len(participants))

    targets = get_target(df, participants)
    s_table = gradient_descent(participant_in, torch.from_numpy(targets), torch.from_numpy(inf_gain))
    torch.save(s_table, "s_table_proj.pt")
    Fs, ths, sds = s_table[:, -1].tolist(), s_table[:, -2].tolist(), s_table[:, -3].tolist()

    # run simulation per participant
    corrects, data = [], []

    for i, p in enumerate(participants):
        d, choices = run_race_model_per_person(df[df["id"] == p], inf_gain, Fs, sds, ths, i)
        data += d
        corrects += (np.array(choices) == df[df["id"] == p]["rating"]).tolist()
    
    ca_rate = 100 * sum(corrects)/len(corrects)
    print(f"Correctness rate: {ca_rate}")

if __name__ == "__main__":
    setup()