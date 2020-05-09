import matplotlib.pyplot as plt
from igraph import plot, Graph


def plot_degree_distribution(g: Graph):
    plt.figure()
    plt.hist(g.degree(), bins=256)
    plt.xlabel("Degree")
    plt.ylabel("#Vertices")
    plt.show()

def plot_big_graph(g: Graph):
    visual_style = {}
    visual_style["bbox"] = (3000,2000)
    visual_style["edge_width"] = [x+0.2 for x in g.es['weight']]
    visual_style["edge_arrow_size"] = .25
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"] = 8
    visual_style["edge_curved"] = False
    visual_style["vertex_label"] = g.vs['name']
    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], niter=10000, grid='nogrid')
    return plot(g, **visual_style)
