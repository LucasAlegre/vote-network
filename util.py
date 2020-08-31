import matplotlib.pyplot as plt
from pyvis.physics import Physics
import seaborn as sns
import numpy as np
import pandas as pd
from igraph import plot, Graph, drawing, summary, mean
from pyvis.network import Network
from datetime import datetime
import time
import plotly.graph_objects as go



sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                            'text.usetex': True,
                            'xtick.labelsize': 16,
                            'ytick.labelsize': 16,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 17,
                            'lines.linewidth' : 2,
                            'lines.markersize' : 6,
                            'legend.fontsize': 15})

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

def get_deputy_html(dep, party, df):
    img = df[df['deputado_nome'] == dep]['deputado_urlFoto'].values[0]
    html = """ <html>
    <body>
    <h2>Partido: {}</h2>
    <img src="{}" width="100" height="150">
    </body>
    </html> """.format(party, img)
    return html

def draw_vis(g: Graph, groups, info=None, parties=None, theme=None, period=None, df=None):
    net = Network(width="100%", height="100%", heading="Câmara dos Deputados")#, bgcolor="#222222", font_color="white")

    labels = g.vs['name']
    for i in g.vs.indices:
        size = 60 if labels[i] in parties else 20
        net.add_node(i, label=labels[i], group=groups[i], title=get_deputy_html(labels[i], info[i], df), borderWidth=2, borderWidthSelected=4, size=size)

    weights = g.es['weight']
    for i, e in enumerate(g.es):
        pair = e.tuple
        net.add_edge(pair[0], pair[1], value=weights[i])

    #net.show_buttons()
    options = """
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
                    "springLength": 500
                }
            }
    }"""
    net.set_options(options)

    if theme is not None:
        filename = "graphs/camaradosdeputados_{}_{}.html".format(theme, period)
    else:
        filename = "graphs/camaradosdeputados_{}.html".format(period)
    net.show(filename)

def filter_edges(edges_list, num_nodes, threshold=None, density=0.1):
    edges, weights = [], []
    if threshold is not None:
        for e in edges_list:
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

def plot_distribution(data, filename=None, xlabel='', ylabel='', kde=False):
    plt.figure()
    sns.distplot(data, kde=kde, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename is not None:
        plt.savefig(f'results/plots/{filename}.pdf')

def plot_similarity_distribution(similarities, weight_threshold):
    plt.figure()
    sns.distplot(similarities, kde=False, color='blue')
    plt.axvline(x=weight_threshold, linestyle='--', color='red')
    plt.ylabel("Number of edges")
    plt.xlabel("Generalized similarity")
    plt.show()

def plot_modularity():
    df = pd.read_csv('results/modularity.csv')
    df_2019 = df[df["inicio"] == "2019-01-01"]
    df_2020 = df[df["inicio"] == "2020-01-01"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x = df_2019['tema'], 
        y = df_2019['valor'],
        name='modularity 2019',
        marker_color='indianred',
        text=df_2019['valor'].round(2).tolist(),
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x = df_2020['tema'], 
        y = df_2020['valor'],
        name='modularity 2020',
        marker_color='lightsalmon',
        text=df_2020['valor'].round(2).tolist(),
        textposition='auto'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)


    fig.update_layout(title='Modularity',
                    plot_bgcolor='rgb(230, 230,230)',
                    showlegend=True)

    fig.show()
    fig.write_image("results/modularity.png", engine="kaleido")

    # import pandas as pd

    # import plotly.graph_objects

    # path = "results/modularity.csv"
    # df = pd.read_csv(path)

    # fig = go.Figure(go.Scatter(x = df["tema"], y = df["valor"],
    #                     name="modularity"))

    # fig.update_layout(title="Modularity per theme",
    #                     plot_bgcolor="rgb(230, 230, 230)",
    #                     showlegend=True)
    # fig.show()

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


def metrics(g: Graph, file):
    summary(g)

    degrees = g.degree()
    betweenness = g.betweenness()
    closeness = g.closeness()

    file.write("\nGLOBAL MEASUURES\n")
    file.write("Connected: " + str(g.is_connected()) + "\n")
    file.write("Density: " + str(g.density()) + "\n")
    file.write("Diameter: " + str(g.diameter()) + "\n")
    file.write("Clustering Coefficient: " + str(g.transitivity_undirected()) + "\n")
    file.write("Average Local Clustering Coefficient: " + str(g.transitivity_avglocal_undirected()) + "\n")
    file.write("Average Degree: " + str(mean(degrees)) + "\n")
    file.write("Max Degree: " + str(g.maxdegree()) + "\n")
    file.write("Average Betweenness: " + str(mean(g.betweenness())) + "\n")
    file.write("Max Betweenness: " + str(max(betweenness)) + "\n")
    file.write("Average Closeness: " + str(mean(closeness)) + "\n")
    file.write("Max Closeness: " + str(max(closeness)) + "\n")

    if not g.is_directed():
        clustering_coef = g.transitivity_local_undirected()

    file.write("\nLOCAL MEASURES\n")
    file.write("Vertex with highest degree: " + str(g.vs.select(_degree = g.maxdegree())['name']) + "\n")
    file.write("Vertex with highest betweenness: " + str(g.vs.select(_betweenness = max(betweenness))['name']) + "\n")
    file.write("Vertex with highest closeness: " + str(g.vs.select(_closeness = max(closeness))['name']) + "\n")
    if not g.is_directed():
        file.write("Vertex with highest clustering coefficient: " + str(g.vs[clustering_coef.index(max(clustering_coef))]['name']) + "\n")


def collect_metrics(g: Graph, parameters):
    node_limit, detection, weight_threshold, density, measure, start_date, end_date, theme, plot_network = parameters

    ts = str(datetime.now())
    file = open("results/metrics_" + str(ts), "w") 

    file.write("Experiment Parameters\n")
    file.write("Node limit: " + str(node_limit) + "\n")
    file.write("Detection alg: " + str(detection) + "\n")
    file.write("Weight Threshold: " + str(weight_threshold) + "\n")
    file.write("Density: " + str(density) + "\n")
    file.write("Measure: " + str(measure) + "\n")

    metrics(g, file)

    file.close()


def get_votations_theme(theme: str, start: str, end: str) -> list:
    with open("resources/votacoes_{}_{}_to_{}.txt".format(theme, start, end), 'r') as file:
        return [v.replace("\n","") for v in file.readlines()]

def get_votations_theme_period_sample(theme, year):
    return np.load("resources/samples/{}_{}_sample.npy".format(theme, year)).tolist()