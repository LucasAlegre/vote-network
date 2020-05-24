#%%
import pandas as pd
import numpy as np
from igraph import Graph, plot, summary
from itertools import combinations
from collections import Counter
from util import plot_big_graph, plot_weight_distribution, set_color_from_communities, draw_vis
import matplotlib.pyplot as plt


#%% Read data
df = pd.read_csv('resources/votos_31-01-2019_to_30-12-2020.csv')

# Remove votations with less than 10 votes
#counts = df['idVotacao'].value_counts()
#df = df[~df['idVotacao'].isin(counts[counts < 10].index)]

num_votations = df['idVotacao'].nunique()
print('Número de votações:', num_votations)

#%% Deputados
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

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > 0.6], weights=True)
summary(g)
# Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]

communities = g.community_multilevel(weights='weight')
draw_vis(g, communities)
#plot_big_graph(g)


# Partidos

""" parties = df['deputado_siglaPartido'].unique()
dep_to_ind = {parties[i]: i for i in range(len(parties))}
votations = df['idVotacao'].unique()
votation_to_ind = {votations[i]: i for i in range(len(votations))}

m = np.zeros((len(parties), len(votations)))
df_grouped = df.groupby(['idVotacao', 'deputado_siglaPartido'])
for group, df_group in df_grouped:
    i = dep_to_ind[group[1]]
    j = votation_to_ind[group[0]]
    m[i,j] = len(df_group[df_group['voto'] == "Sim"]) - len(df_group[df_group['voto'] == "Não"])

def correlation(m, i, j):
    a = m[i,:] - np.mean(m[i,:])
    b = m[j,:] - np.mean(m[j,:])
    return np.dot(a,b) / (np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b)))

edges = dict()
for p1, p2 in combinations([i for i in range(len(parties))], 2):
    edges[(parties[p1], parties[p2])] = correlation(m, p1, p2)

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > 0.0], weights=True)
summary(g)
# Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]

communities = g.community_multilevel(weights='weight')
#set_color_from_communities(g, communities)
draw_vis(g, communities)
#plot_big_graph(g, size=(1500,1000), vertex_size=50) """
