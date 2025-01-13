#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:02:15 2024

@author: kaveen-prabodhya
"""

# Apriori Algorythm

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

print(dataset.describe())
print(dataset.describe(include='object'))
print(dataset.info())


transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1]) if pd.notnull(dataset.values[i, j])])

# Training Apriori on the dataset
from apyori import apriori

# min support -> let's say we look for three product that sold each day for whole week
# and there are total 7500 transactions, so min-sup = 3*7/7500

# in confidence 0.2 means 20% and when confidence is high it can give irrelevent combination
# like customer bought water and egg together it may be because weather impact 
# (becuae of rough weather high temparature people by water but also egg doesnt sense)
# so that combination is absurb
# in 0.8 confidence we say each rule should be correct at least 80% of transaction
# means rule must be true at least 4 time out of five (if we use rule five times 4 times it should true) 
# there is no rule true at least 4 out of 5
# when that confidence divide by 2 and make 0.4 it give some relvent rules but
# further dividing it by 2 and we got 0.2 confidence is chosen becuase 
# those are the most relevent rules or combination choices that we need
# we have lot of transaction and lot of transaction so we need a lower confidence
# we can try different values for minimum lift and here we have have minimum lift of 3
# it shows the relevence of the rule

"""
measures the strength of a rule by evaluating how much more likely the items in the consequent 
are to be purchased when the antecedent occurs compared to when the antecedent does not occur. 
In simpler terms, lift tells you whether the occurrence of one set of items positively, negatively, 
or independently affects the likelihood of the occurrence of another set of items.
"""

# confidence is higher that 40% means the basket contain product that most purchased in the store
# the product is most purchased one and has a grethar than 40% of percent means higher 0.4 confidence
# so its a product that highly bought comapre to all transactions
# doesn't mean it goes well with the relation like buying choclate and beef
# because both are higher than 40% of total transaction but doesn't make sense
# because thay were in someone's basket contain 
# and we sure take it not as a rule

# we can change the confidence the nwe can change the support 
# here our support is consider three products a day it can be four products a day

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)

# Convert dataset to a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Calculate item frequencies
item_counts = df.sum().sort_values(ascending=False)

# Specify the top N items to plot (equivalent to nTree in arules)
top_n = 20  # Change to the desired number, e.g., 20 or 100
top_item_counts = item_counts.head(top_n)

# Plot using matplotlib
item_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Item Frequency Plot (Matplotlib)')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot the top N items using matplotlib
top_item_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title(f'Top {top_n} Item Frequency Plot (Matplotlib)')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Convert the results into a readable DataFrame
def extract_apriori_results(results):
    records = []
    for result in results:
        for ordered_stat in result.ordered_statistics:
            record = {
                'Antecedents': tuple(ordered_stat.items_base),
                'Consequents': tuple(ordered_stat.items_add),
                'Support': result.support,
                'Confidence': ordered_stat.confidence,
                'Lift': ordered_stat.lift
            }
            records.append(record)
    
    return pd.DataFrame(records)

# Extract the results into a DataFrame
rules_df = extract_apriori_results(results)

# Display the DataFrame
print(rules_df)

# Sort the rules by Lift and select the top 10
top_rules = rules_df.sort_values(by='Lift', ascending=False).head(10)
top_rules['Antecedents'] = top_rules['Antecedents'].astype(str)
top_rules['Consequents'] = top_rules['Consequents'].astype(str)

# Plot the top rules
plt.figure(figsize=(12, 8))  # Increased figure size
sns.barplot(x='Lift', y='Antecedents', data=top_rules, hue='Consequents', dodge=False, palette='pastel')

plt.title('Top 10 Association Rules by Lift')
plt.xlabel('Lift')
plt.ylabel('Antecedents')

# Move legend outside the plot
plt.legend(title='Consequents', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Automatically adjusts subplot parameters to give space for labels
plt.show()

#%matplotlib auto
# Plot Confidence vs Lift
scatter = sns.scatterplot(
    x='Confidence',
    y='Lift',
    size='Support',
    hue='Antecedents',
    data=rules_df,
    palette='Set2',
    sizes=(25, 200),  # Reduced size range for points (smaller points)
    alpha=0.7  # Add transparency to points
)

plt.title('Confidence vs Lift')
plt.xlabel('Confidence')
plt.ylabel('Lift')

# Remove the default legend to avoid repetition
handles, labels = scatter.get_legend_handles_labels()

# Customizing the legend without repetition
plt.legend(
    handles=handles[:len(set(labels))],  # Ensure no duplicates by selecting unique labels
    labels=list(set(labels)),  # Set unique labels
    title='Antecedents', 
    bbox_to_anchor=(1, 1), 
    loc='upper left', 
    borderaxespad=0., 
    fontsize='small', 
    ncol=3
)

# Ensure layout is adjusted to fit everything
plt.subplots_adjust(right=0.85)
plt.tight_layout(pad=3.0)

# Show the plot
plt.show()
