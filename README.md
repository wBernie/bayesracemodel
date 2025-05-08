# A Bayesian Race Model for Number Generalization
We examine whether Bayesian inference on A Large Dataset of Generalization Patterns in the Number Game (Bigelow & Piantadosi, 2016) sufficiently explains both group-level and participant-level behaviour on the task through a lognormal race model

## Running
Use _bayesracemodel.py_ to reproduce results

## Model
 Priors were derived from aggregated qualitative descriptions of the rule sets provided by participants. Likelihoods were calculated as the ratio of stimuli that fit a concept relative to the size of the stimuli set. The posterior probability was proportional to the product of the prior and likelihood. This allowed an evaluation of how well each potential rule explained the observed responses:
 <p align=center>
 $P(C_{j}|S_{i}^{k}) \propto P(S_{i}^{k}|C_j) * P(C_j)$<br>
   where $C_j$ (1 <= j <= 101) is a concept, $t_{S,i}^{k}$ is the target presented to participant i when presented with set $S_i$,
 </p>
     <br>
If the participant’s response was “yes”:
     <p align=center>
       $S_{i}^{k}$: $S_i = S_i^{k-1} \cup t_{S,i}$ (2)
     </p>   
If the participant's response was "no" then we simply have the same set: 
     <p align=center>
     $S_i^k = S_i^{k-1}$
     </p>
     
### Race Model
<p align=center>
  $RT_{i}^{k} = F_i * exp(- max(s_1, s_2, …., s_j,..., s_101, th_i + sd_i))$<br>
  where given a set S, $RT_{i}^{k}$ is the response of participant i to the kth number presented for a rating in S, $F_i$ is the participant-specific latency factor, $th_i$ is the participant-specific decision threshold, $sd_i$ is the participant-specific standard deviation of zero-mean normally distributed noise for simulating choice, and $s_j$ is scored for a given concept $C_j$.
</p>
The scores used in the model were one of two mechanisms used to guide the selection process: probability strength 
     <p align=center>
       $logit(P(C_k|S_i))$ (used commonly within ACT-R)
     </p> 
and information gain 
     <p align=center>
       $log(P(C_j|S_{i}^{k})/P(C_j|S_{i-1}^k))$ (as justified for production rules in Clarion (Sun, 2016))
     </p> 

## Results
### Group Results
<p align="center">
 <img width="50%" src="https://github.com/wBernie/bayesracemodel/blob/main/bayesraceimages/groupresults.png"><br>
 Bar plot of per-set accuracy for all 255 sets, in no particular order.<br>
</p>
With the priors used, we found an accuracy of 64% with a variance of 0.8. For all sets, the best concept found was either even numbers, odd numbers, or numbers between the largest and smallest numbers in the set. Specifically, if the set contained only even or only odd numbers, the best concept found was even numbers or odd numbers respectively. Otherwise, the best concept found was the numbers between the largest and smallest numbers in the set.


### Logit Model
<p align="center">
 <img width="50%" src="https://github.com/wBernie/bayesracemodel/blob/main/bayesraceimages/resultslogit.png"><br>
 Bar plot of “yes” responses to targets from race model with logit scores compared to humans for a given set<br>
</p>
The performance metric of interest is the percentage of simulated “yes”/”no” answers that align with the participant responses, which is equal to the sum of generated answers that were the same as the participant responses averaged over the total number of responses. The maximum achieved performance was ~47.09% (see Figure 2 for a representative sample) using the logit of the posterior as the statistic for strength. We then reattempted the simulation using the log of the information gain and found an accuracy of ~76.76% (see Figure 3 for a representative sample). Given that 31% of responses are “no”, the first approach scores lower due to significantly higher probability of saying yes to a target for any given set when compared to the second approach (see Figures 2 and 3 for an example). This low variation in responses suggests a worse fit compared to the second approach.

### I.G Model
<p align="center">
 <img width="50%" src="https://github.com/wBernie/bayesracemodel/blob/main/bayesraceimages/resultsig.png"><br>
 Bar plot of “yes” responses to targets from the race model with information-gain scores compared to humans for a given set
  <br>
</p>
The second approach shows sufficient variation in its responses, and also beats a reasonable baseline of 69% (saying “no” to every response), given that 31% of human responses in the 272,700 trials are “yes” by 7%. Thus, variation in the latter approach is indicative of a good fit for response choices.
