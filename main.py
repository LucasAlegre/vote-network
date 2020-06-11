# %%
import argparse
import ntpath
import random

import pandas as pd
import numpy as np
from igraph import Graph, plot, summary
from itertools import combinations
from collections import Counter
from util import plot_big_graph, plot_weight_distribution, set_color_from_communities, draw_vis
import matplotlib.pyplot as plt


def correlation(m, i, j):
    a = m[i, :] - np.mean(m[i, :])
    b = m[j, :] - np.mean(m[j, :])
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def main():
    parser = argparse.ArgumentParser(description="Community detection in voting networks")
    parser.add_argument('-s', '--sample', type=int,
                        action="store", dest="node_limit", default=None,
                        help="Sample a random subset of representatives of size up-to N")
    parser.add_argument('-c', '--correlation', type=float,
                        action="store", dest="min_correlation", default=0.6,
                        help="Minimum correlation to establish an edge between nodes, (0, 1]")
    parser.add_argument("-a", "--algorithm", choices=["spinglass", "multilevel"],
                        action="store", dest="community_alg", default="multilevel",
                        help="Choice of community detection algorithm")
    args = parser.parse_args()

    node_limit = args.node_limit
    detection = args.community_alg
    weight_threshold = args.min_correlation

    print("Sample limit: {}".format(node_limit))
    print("Community detection: {}".format(detection))
    print("Edge weight threshold: {}".format(weight_threshold))

    # %% Read data
    path = 'resources/votos_31-01-2019_to_10-05-2020.csv'
    df = pd.read_csv(path)

    basename = ntpath.basename(path)
    print(basename)
    random.seed(0)

    all_reps = df['deputado_nome'].unique()
    total_reps = len(all_reps)
    num_reps = total_reps

    if node_limit is not None:
        print("Randomly sampling at most {} representatives".format(node_limit))
        sample_size = min(total_reps, node_limit)
        num_reps = sample_size
        reps = random.sample(list(all_reps), sample_size)

        # Mask to select only those reps that were sampled
        mask = df['deputado_nome'].apply(lambda x: x in reps)
        # Filter out rows for non-sampled reps
        df = df[mask]
    else:
        print("Using data from all {} representatives.".format(total_reps))
        reps = all_reps

    print(df.head(n=5))

    # Remove motions with less than 10 votes
    # counts = df['idVotacao'].value_counts()
    # df = df[~df['idVotacao'].isin(counts[counts < 10].index)]

    num_motions = df['idVotacao'].nunique()
    print('Número de votações:', num_motions)
    print("Building graph for {} reps and {} voting motions.".format(num_reps, num_motions))

    rep_to_ind = {reps[i]: i for i in range(len(reps))}

    motions = df['idVotacao'].unique()
    motion_to_ind = {motions[i]: i for i in range(len(motions))}

    parties = [p for p in df['deputado_siglaPartido'].unique() if pd.notna(p)]

    edges = dict()

    cor_matrix = np.zeros((len(reps), len(motions)))
    df_grouped = df.groupby(['idVotacao', 'deputado_nome'])
    for group, df_group in df_grouped:
        voto = df_group['voto'].values[0]
        i = rep_to_ind[group[1]]
        j = motion_to_ind[group[0]]
        if voto == "Sim":
            cor_matrix[i, j] = 1
        if voto == "Não":
            cor_matrix[i, j] = -1

    for rep1, rep2 in combinations([i for i in range(len(reps))], 2):
        edges[(reps[rep1], reps[rep2])] = correlation(cor_matrix, rep1, rep2)

    for group, df_group in df.groupby('deputado_nome'):
        partidos = {p for p in df_group['deputado_siglaPartido'].values if pd.notna(p)}
        for p in partidos:
            edges[(group, p)] = 1.0

    # Partidos
    party_to_ind = {parties[i]: i for i in range(len(parties))}
    motions = df['idVotacao'].unique()
    motion_to_ind = {motions[i]: i for i in range(len(motions))}

    cor_matrix = np.zeros((len(parties), len(motions)))
    df_grouped = df.groupby(['idVotacao', 'deputado_siglaPartido'])
    for group, df_group in df_grouped:
        i = party_to_ind[group[1]]
        j = motion_to_ind[group[0]]
        cor_matrix[i, j] = len(df_group[df_group['voto'] == "Sim"]) - len(df_group[df_group['voto'] == "Não"])

    for p1, p2 in combinations([i for i in range(len(parties))], 2):
        edges[(parties[p1], parties[p2])] = correlation(cor_matrix, p1, p2)

    g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > weight_threshold], weights=True)
    summary(g)
    # Normalize weights to [0,1]
    maxw = max(g.es['weight'])
    minw = min(g.es['weight'])
    g.es['weight'] = [(e - minw) / (maxw - minw) for e in g.es['weight']]

    if detection == 'spinglass':
        communities = g.community_spinglass(weights='weight', spins=3)
    elif detection == 'multilevel':
        communities = g.community_multilevel(weights='weight')
    else:
        return

    draw_vis(g, communities, parties)


if __name__ == "__main__":
    main()
