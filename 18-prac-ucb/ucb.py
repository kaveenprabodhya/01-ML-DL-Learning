#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:22:17 2024

@author: kaveen-prabodhya
"""

# Upper Confidence Bound

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selelcted = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_rewards = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i]/ number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selelcted.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_rewards += reward
    
# Visualising the results
plt.hist(ads_selelcted)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()