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


def filter_by_theme(df, theme, start_date, end_date):
    print("Filtering votations by theme: " + theme)
    subject_votations = get_votations_theme(theme, start_date, end_date)
    print(subject_votations)
    
    mask = df['idVotacao'].apply(lambda x: x in subject_votations)
    return df[mask]

def filter_by_name_and_quantity(df, node_limit):

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

    num_motions = df['idVotacao'].nunique()

    print("Building graph for {} reps and {} voting motions.".format(num_reps, num_motions))

    return df, reps

def save_modularity(modularity, theme, start_date, end_date):
    with open("results/modularity.csv", "a") as file:
        file.write("{},{},{},{}\n".format(theme, start_date, end_date, modularity))

    
def get_params():
    parser = argparse.ArgumentParser(description="Community detection in voting networks")
    parser.add_argument('-s', "--sample", type=int,
                        action="store", dest="node_limit", default=None,
                        help="Sample a random subset of representatives of size up-to N")
    parser.add_argument('-c', '--correlation', type=float,  #0.9998
                        action="store", dest="min_correlation", default=None,
                        help="Minimum correlation to establish an edge between nodes, [0, 1]")
    parser.add_argument('-d', "--density", type=float, action="store", dest="density", default=0.3,
                        help="Desired density to filter edges")
    parser.add_argument('-m', '--measure', dest="measure", action="store", choices=["pearson", "generalized"],
                        default="generalized", help="Similarity measure to create edges")
    parser.add_argument("-a", "--algorithm", choices=["leiden", "spinglass", "multilevel", "party"],
                        action="store", dest="community_alg", default="leiden",
                        help="Choice of community detection algorithm")
    parser.add_argument("-p", "--plot", dest="plot_network", action="store_true", default=False, 
                        help="Plot the network's graph (Y/N)")

    parser.add_argument('-b', "--begin", type=str,
                        action="store", dest="start_date", default="2019-01-31",
                        help="YYYY-MM-DD  Start data for the period that you want to collect the data.")
    parser.add_argument('-e', "--end", type=str,
                        action="store", dest="end_date", default="2020-12-30",
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
    plot_network = args.plot_network
    start_date = args.start_date
    end_date = args.end_date
    theme = args.proposition_themes

    return node_limit, detection, weight_threshold, density, measure, start_date, end_date, theme, plot_network

def main():
    
    node_limit, detection, weight_threshold, density, measure, start_date, end_date, theme, plot_network = get_params()
    experiment_parameters = (get_params())

    print("Sample limit: {}".format(node_limit))
    print("Community detection: {}".format(detection))
    print("Edge weight threshold: {}".format(weight_threshold))

    # %% Read data
    path = 'resources/votos_31-01-2019_to_30-12-2020.csv'
    df = pd.read_csv(path)

    basename = ntpath.basename(path)
    print(basename)
    random.seed(0)

    if theme is not None:
        df = filter_by_theme(df, theme, start_date, end_date)
    df, reps = filter_by_name_and_quantity(df, node_limit)

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
    
    plot_similarity_distribution([e[1] for e in edges if e[1] > 0.99], weight_threshold)

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
    g.save('graphs/g.graphml')

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
        
    modularity = g.modularity(communities, 'weight')
    print("Modularity Score: ", modularity)
    save_modularity(modularity, theme, start_date, end_date)

    info = [parties[i] for i in groups_by_party(df, reps, parties)]
    
    period = start_date + '_to_' + end_date

    if plot_network:
        draw_vis(g, groups=communities, info=info, parties=parties, theme=theme, period=period)

    collect_metrics(g, experiment_parameters)


if __name__ == "__main__":
    #g = read('g.graphml')
    #plot_distribution(g.betweenness(), filename='betweenness', xlabel='Betweenness', ylabel='Number of edges')
    #draw_vis(g)
    main()