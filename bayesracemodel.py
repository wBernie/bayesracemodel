from pyClarion import Process, NumDict, Index, Event, numdict, bind, UpdateSite, KeyForm, Key, path, root, Agent
from pyClarion.knowledge import Family, Atoms, Atom
import math
from datetime import timedelta
from typing import *

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
    input: NumDict

# Family, Sort, and Terms are a hierarchy

    def __init__(self, 
                name:str, pfam:Family, index: Index, F: float = 1, sd: float=1.0) -> None:
        super().__init__(name)
        self.p = type(self).Params()
        pfam[name] =  self.p# assign the keyspace to this attr, under the processes own name
        # the params under p are now under p -> name -> params
        self.choice = numdict(index, {}, 0.0) #default 0 in the beginnning, 
        self.sample = numdict(index, {}, 0.0)
        self.input = numdict(index, {}, 0.0)
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
        sample = self.input.normalvariate(numdict(self.input.i, {}, sd))
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

#TODO: function to fit parameters:
#TODO: function to run simulation 
#TODO: both of these will be done once the likelihoods and priors format is determined. 

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

    return df["count"].to_numpy()[np.newaxis, :]


