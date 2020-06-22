import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from igraph import plot, Graph, drawing
from pyvis.network import Network


def pearson_correlation(m):
    arcs0 = m - m.mean(axis=1)[:, np.newaxis]
    arcs1 = m.T - m.mean(axis=0)[:, np.newaxis]
    M = arcs0.dot(arcs0.T)
    m = np.sqrt(M.diagonal())
    M = ((M / m).T / m).T
    return M

def generalized_similarity(m, min_eps=0.001, max_iter=1000):
    """ Balázs Kovács, "A generalized model of relational similarity," Social Networks, 32(3), July 2010, pp. 197–211
        Based on: https://github.com/dzinoviev/generalizedsimilarity
    """
    arcs0 = m - m.mean(axis=1)[:, np.newaxis]
    arcs1 = m.T - m.mean(axis=0)[:, np.newaxis]

    eps = min_eps + 1
    N = np.eye(m.shape[1])

    iters = 0
    while (eps > min_eps and iters < max_iter) or np.isnan(N).any():
        M = arcs0.dot(N).dot(arcs0.T)
        m = np.sqrt(M.diagonal())
        M = ((M / (m+1e-8)).T / (m+1e-8)).T
        
        Np = arcs1.dot(M).dot(arcs1.T)
        n = np.sqrt(Np.diagonal())
        Np = ((Np / (n+1e-8)).T / (n+1e-8)).T
        eps = np.abs(Np - N).max()
        N = Np

        iters += 1
    
    return M

def draw_vis(g: Graph, groups, info=None, parties=None):
    net = Network(width="100%", height="100%")#, bgcolor="#222222", font_color="white")

    labels = g.vs['name']
    for i in g.vs.indices:
        size = 60 if labels[i] in parties else 20
        net.add_node(i, label=labels[i], group=groups[i], title=info[i], borderWidth=2, borderWidthSelected=4, size=size)
        
    weights = g.es['weight']
    for i, e in enumerate(g.es):
        pair = e.tuple
        net.add_edge(pair[0], pair[1], value=weights[i])

    #net.show_buttons()
    net.set_options("""
        var options = {
              "nodes": {
                "font": {
                    "size": 30
                }
            },
            "edges": {
                "color": {
                    "inherit": true
                },    
                "scaling": {
                    "min": 0,
                    "max": 1
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -80000,
                    "springConstant": 0.001, 
                    "springLength": 300
                }
            }
    }""")
    net.show("mygraph.html")

def filter_edges(edges_list, num_nodes, threshold=None, density=0.1):
    edges, weights = [], []
    if threshold is not None:
        for e in range(len(edges_list)):
            if e[1] >= threshold:
                edges.append(e[0])
                weights.append(e[1])
    else:
        count = int(num_nodes * (num_nodes - 1) * density / 2)
        edges_list.sort(reverse=True, key=lambda e: e[1])
        edges_list = edges_list[:count]
        edges = [e[0] for e in edges_list]
        weights = [e[1] for e in edges_list]
    return edges, weights

def groups_by_party(df, reps, parties):
    rep_to_party = {}
    parties = parties + ['Sem Partido']
    for group, df_group in df.groupby('deputado_nome'):
        ps = [p for p in df_group['deputado_siglaPartido'] if pd.notna(p)]
        rep_to_party[group] = ps[0] if len(ps) > 0 else 'Sem Partido'
    return [parties.index(rep_to_party[rep]) for rep in reps]

def set_color_from_communities(g: Graph, communities):
    pal = drawing.colors.ClusterColoringPalette(len(communities))
    g.vs['color'] = pal.get_many(communities.membership)

def plot_weight_distribution(g: Graph):
    plt.figure()
    plt.hist(g.es['weight'], bins=256)
    plt.xlabel("Weight")
    plt.ylabel("#Edges")
    plt.show()

def plot_degree_distribution(g: Graph):
    plt.figure()
    plt.hist(g.degree(), bins=256)
    plt.xlabel("Degree")
    plt.ylabel("#Vertices")
    plt.show()

def plot_big_graph(g: Graph, size=(3000,2000), vertex_size=20):
    visual_style = {}
    visual_style["bbox"] = size
    visual_style["edge_width"] = [x**2 + 1 for x in g.es['weight']]
    visual_style["edge_arrow_size"] = .25
    visual_style["vertex_size"] = vertex_size
    visual_style["vertex_label_size"] = 20
    visual_style["edge_curved"] = False
    visual_style["vertex_label"] = g.vs['name']
    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], grid='nogrid')
    return plot(g, **visual_style)
