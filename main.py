#%%
import pandas as pd
from igraph import Graph, plot, summary
from itertools import combinations
from collections import Counter
from util import plot_big_graph
import matplotlib.pyplot as plt


#%%
df = pd.read_csv('resources/votos_01-02-2015_to_31-01-2019.csv')

# Remove votations with less than 10 votes
counts = df['idVotacao'].value_counts()
df = df[~df['idVotacao'].isin(counts[counts < 10].index)]

num_votations = df['idVotacao'].nunique()
print('Número de votações:', num_votations)

#%%
edges = Counter()
df_grouped = df.groupby(['idVotacao', 'voto'])
for group, df_group in df_grouped:
    for dep1, dep2 in combinations(df_group['deputado_nome'], 2):
        edges[tuple(sorted((dep1,dep2)))] += 1

g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight/num_votations > 0.5], weights=True)
summary(g)

#%% Normalize weights to [0,1]
maxw = max(g.es['weight'])
minw = min(g.es['weight'])
g.es['weight'] = [(e-minw)/(maxw-minw) for e in g.es['weight']]

#%%
plot_big_graph(g)
