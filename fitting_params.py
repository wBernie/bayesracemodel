import torch
import math 
from torch import erfc
from typing import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def forward_pass(inputs, weights):
    return inputs @ weights

def cost(inputs, targets, weights, concept_scores):

    outs = forward_pass(inputs, weights)
    subject_outs = outs

    F, th, sd = subject_outs[:, -1].unsqueeze(-1), subject_outs[:, -2].unsqueeze(-1), subject_outs[:, -3].unsqueeze(-1) 
    #sd must be positive:
    sd = sd ** 2 # the NN will adjust to make this a sqrt
    return log_likelihood(targets[:, 0], concept_scores, targets[:, 1], F, sd, th)

def log_likelihood(rts: torch.tensor, concept_scores: torch.tensor, choice: torch.tensor, F_i: float, sd_i: float, th_i:float) -> torch.tensor:

    #unsqueeze somethings to match dimensions:
    rts = rts.unsqueeze(1).unsqueeze(1) # 606x1x1 as comapred to 606x15x30x101
    # renaming
    target = rts
    sd = sd_i
    F = F_i
    mean_nochoice = -th_i + torch.log(F)

    z = math.sqrt(6)*sd/math.pi
    m = torch.log(torch.exp(concept_scores/z).sum(axis=-1, keepdim=True))*z #here we have 606x15x30
    mu = 1/z
    beta = torch.exp((torch.log(F) - m)/z)

    l_word = torch.log(mu) - torch.log(beta) + (mu-1)* torch.log(target) - ((target)**mu)/beta + \
        torch.log(0.5*erfc((torch.log(target) - mean_nochoice)/ sd))
    
    l_nword = (-torch.log(target * sd * math.sqrt(2* math.pi)) - 
               ((torch.log(target) - mean_nochoice)**2) / (2 * sd**2) - (target/beta)**mu)
    
    likelihood = torch.where(choice.bool(), l_word, l_nword)
    likelihood = torch.where((-torch.inf < likelihood) & (likelihood < torch.inf), likelihood, torch.nan)
    return torch.nansum(likelihood)

def backward_pass(inputs, targets, weights, concept_scores):
    x = cost(inputs, targets, weights, concept_scores)
    return x

def gradient_descent(inputs, targets, concept_scores, eta=1e-3, iterations=2500):
    #split the df into the different subjects experiments
    sub_table = torch.rand((inputs.shape[0], 3)) # 3 subject specific parameters 
    sub_table = sub_table.clone().detach().requires_grad_(True)

    cost_over_time = []

    pbar = tqdm(range(iterations))

    for k in pbar:
        c = backward_pass(inputs, targets, sub_table, concept_scores)

        c.backward()

        # Store gradients
        sub_grad = sub_table.grad.clone()
        
        # print(sub_grad)

        # Zero gradients for next iteration
        sub_table.grad.zero_()

        with torch.no_grad():
            sub_table.data += eta * sub_grad

        if k != 0: cost_over_time.append(c.item())
        pbar.set_description(f"Loss: {c}")

        #plot cost over time
        if k != 0:
            plt.plot(list(range(k)), cost_over_time)
            plt.savefig("cost.png")
            plt.clf()

    return sub_table