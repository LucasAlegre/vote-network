import matplotlib.pyplot as plt
from igraph import plot, Graph, drawing


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
