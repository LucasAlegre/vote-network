# %%
import argparse
import ntpath
import random
import pandas as pd
import numpy as np
from igraph import Graph, plot, summary, read
from itertools import combinations
from collections import Counter
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
import leidenalg


def main():
    parser = argparse.ArgumentParser(description="Community detection in voting networks")
    parser.add_argument('-s', "--sample", type=int,
                        action="store", dest="node_limit", default=None,
                        help="Sample a random subset of representatives of size up-to N")
    parser.add_argument('-c', '--correlation', type=float,
                        action="store", dest="min_correlation", default=None,
                        help="Minimum correlation to establish an edge between nodes, [0, 1]")
    parser.add_argument('-d', "--density", type=float, action="store", dest="density", default=0.3,
                        help="Desired density to filter edges")
    parser.add_argument('-m', '--measure', dest="measure", action="store", choices=["pearson", "generalized"],
                        default="generalized", help="Similarity measure to create edges")
    parser.add_argument("-a", "--algorithm", choices=["leiden", "spinglass", "multilevel", "party"],
                        action="store", dest="community_alg", default="leiden",
                        help="Choice of community detection algorithm")
    parser.add_argument("-p", "--plot", choices=["Y", "N"], dest="plot_network", action="store", default="Y", 
                        help="Plot the network's graph (Y/N)")

    parser.add_argument('-b', "--begin", type=str,
                        action="store", dest="start_date", default="2020-01-01",
                        help="YYYY-MM-DD  Start data for the period that you want to collect the data.")
    parser.add_argument('-e', "--end", type=str,
                        action="store", dest="end_date", default=None,
                        help="YYYY-MM-DD  End data for the period that you want to collect the data.")
    parser.add_argument('-t', "--theme", type=str,
                        action="store", dest="proposition_themes", default=None,
                        help="Desired theme for the proposition collect (saude/educacao/economia). Default: None")

    args = parser.parse_args()

    node_limit = args.node_limit
    detection = args.community_alg
    weight_threshold = args.min_correlation
    density = args.density
    measure = args.measure
    plot_network = args.plot_network.upper()

    start_date, end_date, theme = args.start_date, args.end_date, args.proposition_themes

    experiment_parameters = (node_limit, detection, weight_threshold, density, measure)

    print("Sample limit: {}".format(node_limit))
    print("Community detection: {}".format(detection))
    print("Edge weight threshold: {}".format(weight_threshold))

    # %% Read data
    path = 'resources/votos_31-01-2019_to_30-12-2020.csv'
    #path = 'resources/votos_01-02-2015_to_31-01-2019.csv'
    df = pd.read_csv(path)

    basename = ntpath.basename(path)
    print(basename)
    random.seed(0)


    if theme is not None:
        
        print("Filtering votations by theme: " + theme)
        subject_votations = get_votations_theme(theme, start_date, end_date)
        print(subject_votations)
       
        mask = df['idVotacao'].apply(lambda x: x in subject_votations)
        df = df[mask]


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

    print("Building graph for {} reps and {} voting motions.".format(num_reps, num_motions))

    rep_to_ind = {reps[i]: i for i in range(len(reps))}
    motions = df['idVotacao'].unique()
    motion_to_ind = {motions[i]: i for i in range(len(motions))}
    parties = [p for p in df['deputado_siglaPartido'].unique() if pd.notna(p)]
    edges = []

    vote_matrix = np.zeros((len(reps), len(motions)))
    df_grouped = df.groupby(['idVotacao', 'deputado_nome'])
    for group, df_group in df_grouped:
        voto = df_group['voto'].values[0]
        i = rep_to_ind[group[1]]
        j = motion_to_ind[group[0]]
        if voto == "Sim":
            vote_matrix[i,j] = 1
        if voto == "Não":
            vote_matrix[i,j] = -1

    if measure == 'generalized':
        M = generalized_similarity(vote_matrix)
    elif measure == 'pearson':
        M = pearson_correlation(vote_matrix)
    else:
        raise NotImplementedError

    for dep1, dep2 in combinations(range(len(reps)), 2):
        if M[dep1,dep2] > 0:
            edges.append(((dep1,dep2), M[dep1,dep2]))

    #similarities = [e[1] for e in edges if e[1] > 0.9]
    #plot_similarity_distribution(similarities, weight_threshold)

    """ for group, df_group in df.groupby('deputado_nome'):
        partidos = {p for p in df_group['deputado_siglaPartido'].values if pd.notna(p)}
        for p in partidos:
            edges[(group, p)] = 1.0
    """
    # Parties
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

    if measure == 'generalized':
        M = generalized_similarity(vote_matrix)
    elif measure == 'pearson':
        M = pearson_correlation(vote_matrix)
    else:
        raise NotImplementedError
    for p1, p2 in combinations([i for i in range(len(parties))], 2):
        if M[p1,p2] > 0:
            edges.append(((p1, p2), M[p1,p2])) """

    #g = Graph.TupleList([(*pair, weight) for pair, weight in edges.items() if weight > weight_threshold], weights=True)
    g = Graph(graph_attrs={'name': 'Camera dos Deputados'}, directed=False)
    g.add_vertices(reps)
    edges, weights = filter_edges(edges, num_nodes=g.vcount(), threshold=weight_threshold, density=density)
    g.add_edges(edges)
    g.es['weight'] = weights
    # Normalize weights to [0,1]
    maxw = max(g.es['weight'])
    minw = min(g.es['weight'])
    g.es['weight'] = [(e - minw) / (maxw - minw) for e in g.es['weight']]
    summary(g)
    g.save('g.graphml')

    if detection == 'leiden':
        communities = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight', n_iterations=100).membership
        #communities = g.community_leiden(objective_function='modularity', weights='weight', n_iterations=100)
    elif detection == 'spinglass':
        communities = g.community_spinglass(weights='weight').membership
    elif detection == 'multilevel':
        communities = g.community_multilevel(weights='weight').membership
    elif detection == 'party':
        communities = groups_by_party(df, reps, parties)
    else:
        raise NotImplementedError
    print("Modularity Score: ", g.modularity(communities, 'weight'))

    info = [parties[i] for i in groups_by_party(df, reps, parties)]
    
    if (plot_network == "Y"):
        draw_vis(g, groups=communities, info=info, parties=parties)

    collect_metrics(g, experiment_parameters)


if __name__ == "__main__":
    #g = read('g.graphml')
    #plot_distribution(g.betweenness(), filename='betweenness', xlabel='Betweenness', ylabel='Number of edges')
    #draw_vis(g)
    main()