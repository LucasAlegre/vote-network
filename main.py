#%%
import pandas as pd
import numpy as np
from igraph import Graph, plot, summary
from itertools import combinations
from collections import Counter
from util import plot_big_graph, plot_weight_distribution, set_color_from_communities, draw_vis
import matplotlib.pyplot as plt


def pearson_correlation(m):
    arcs0 = m - m.mean(axis=1)[:, np.newaxis]
    arcs1 = m.T - m.mean(axis=0)[:, np.newaxis]
    M = arcs0.dot(arcs0.T)
    m = np.sqrt(M.diagonal())
    M = ((M / m).T / m).T
    return M

def generalized_similarity(m, min_eps=0.001, max_iter=100):
    """ Balázs Kovács, "A generalized model of relational similarity," Social Networks, 32(3), July 2010, pp. 197–211
        Copied and pasted from: https://github.com/dzinoviev/generalizedsimilarity
    """
    arcs0 = m - m.mean(axis=1)[:, np.newaxis]
    arcs1 = m.T - m.mean(axis=0)[:, np.newaxis]

    eps = min_eps + 1
    N = np.eye(m.shape[1])

    iters = 0
    while eps > min_eps and iters < max_iter:
        M = arcs0.dot(N).dot(arcs0.T)
        m = np.sqrt(M.diagonal())
        M = ((M / m).T / m).T
        
        Np = arcs1.dot(M).dot(arcs1.T)
        n = np.sqrt(Np.diagonal())
        Np = ((Np / n).T / n).T
        Np = np.nan_to_num(Np)
        eps = np.abs(Np - N).max()
        N = Np

        iters += 1
    
    return M

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
parties = [p for p in df['deputado_siglaPartido'].unique() if pd.notna(p)]

edges = dict()

vote_matrix = np.zeros((len(deputies), len(votations)))
df_grouped = df.groupby(['idVotacao', 'deputado_nome'])
for group, df_group in df_grouped:
    voto = df_group['voto'].values[0]
    i = dep_to_ind[group[1]]
    j = votation_to_ind[group[0]]
    if voto == "Sim":
        vote_matrix[i,j] = 1
    if voto == "Não":
        vote_matrix[i,j] = -1

M = generalized_similarity(vote_matrix) 
#M = pearson_correlation(vote_matrix)

for dep1, dep2 in combinations([i for i in range(len(deputies))], 2):
    edges[(deputies[dep1], deputies[dep2])] = M[dep1,dep2]

""" for group, df_group in df.groupby('deputado_nome'):
    partidos = {p for p in df_group['deputado_siglaPartido'].values if pd.notna(p)}
    for p in partidos:
        edges[(group, p)] = 1.0
"""
# Partidos
""" party_to_ind = {parties[i]: i for i in range(len(parties))}
votations = df['idVotacao'].unique()
votation_to_ind = {votations[i]: i for i in range(len(votations))}

vote_matrix = np.zeros((len(parties), len(votations)))
df_grouped = df.groupby(['idVotacao', 'deputado_siglaPartido'])
for group, df_group in df_grouped:
    n = len(df_group)
    i = party_to_ind[group[1]]
    j = votation_to_ind[group[0]]
    yes_perc = len(df_group[df_group['voto'] == "Sim"])/n
    if yes_perc < 0.5:
        yes_perc = - (1 - yes_perc)
    vote_matrix[i,j] = yes_perc

M = generalized_similarity(vote_matrix)
#M = pearson_correlation(vote_matrix)
for p1, p2 in combinations([i for i in range(len(parties))], 2):
    edges[(parties[p1], parties[p2])] = M[p1,p2] """

""" plt.figure()
plt.hist(edges.values(), bins=256)
plt.show() """

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > 0.9995], weights=True)
summary(g)
# Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]

""" communities = g.community_spinglass(weights='weight')
print("Modularity Score: ", g.modularity(communities, 'weight'))
communities = g.community_multilevel(weights='weight')
print("Modularity Score: ", g.modularity(communities, 'weight')) """
communities = g.community_leiden(objective_function='modularity', weights='weight', n_iterations=100)
print("Modularity Score: ", g.modularity(communities, 'weight'))
draw_vis(g, communities, parties)
