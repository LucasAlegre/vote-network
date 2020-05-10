#%%
import pandas as pd
import numpy as np
from igraph import Graph, plot, summary
from itertools import combinations
from collections import Counter
from util import plot_big_graph, plot_weight_distribution
import matplotlib.pyplot as plt


#%%
df = pd.read_csv('resources/votos_01-02-2015_to_31-01-2019.csv')

# Remove votations with less than 10 votes
counts = df['idVotacao'].value_counts()
df = df[~df['idVotacao'].isin(counts[counts < 10].index)]

num_votations = df['idVotacao'].nunique()
print('Número de votações:', num_votations)

#%%
deputies = df['deputado_nome'].unique()
dep_to_ind = {deputies[i]: i for i in range(len(deputies))}
votations = df['idVotacao'].unique()
votation_to_ind = {votations[i]: i for i in range(len(votations))}

m = np.zeros((len(deputies), len(votations)))
df_grouped = df.groupby(['idVotacao', 'deputado_nome'])
for group, df_group in df_grouped:
    voto = df_group['voto'].values[0]
    i = dep_to_ind[group[1]]
    j = votation_to_ind[group[0]]
    if voto == "Sim":
        m[i,j] = 1
    if voto == "Não":
        m[i,j] = -1

def correlation(m, i, j):
    a = m[i,:] - np.mean(m[i,:])
    b = m[j,:] - np.mean(m[j,:])
    return np.dot(a,b) / (np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b)))

edges = dict()
for dep1, dep2 in combinations([i for i in range(len(deputies))], 2):
    edges[(deputies[dep1], deputies[dep2])] = correlation(m, dep1, dep2)

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > 0.5], weights=True)
summary(g)
# Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]

""" edges = dict()
for dep1, dep2 in combinations([i for i in range(len(deputies))], 2):
    both_voted_mask = (m[dep1,:] != 0) & (m[dep2,:] != 0)
    both_voted = np.count_nonzero(both_voted_mask)
    votes_equal = np.count_nonzero((m[dep1,:] == m[dep2,:]) & both_voted_mask)
    print(votes_equal, both_voted)
    edges[(deputies[dep1], deputies[dep2])] = votes_equal / both_voted if both_voted/num_votations > 0.33 else 0

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > 0.85], weights=True)
# Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]
summary(g) """

#plot_weight_distribution(g)
plot_big_graph(g)
