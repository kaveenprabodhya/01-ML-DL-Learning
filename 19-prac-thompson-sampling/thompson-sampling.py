#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:21:00 2024

@author: kaveen-prabodhya
"""

# Thompson Sampling

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selelcted = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_rewards = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
       random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
       if random_beta > max_random:
           max_random = random_beta
           ad = i
    ads_selelcted.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    total_rewards += reward
    
# Visualising the results
plt.hist(ads_selelcted)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()