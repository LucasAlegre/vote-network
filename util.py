import matplotlib.pyplot as plt
from igraph import plot, Graph, drawing
from pyvis.network import Network


def draw_vis(g: Graph, communities=None):
    net = Network(width="100%", height="100%")#, bgcolor="#222222", font_color="white")
    net.barnes_hut()

    labels = g.vs['name']
    if communities is not None:
        groups = communities.membership
        for i in g.vs.indices:
            net.add_node(i, label=labels[i], group=groups[i], borderWidth=2, borderWidthSelected=4)
    else:
        for i in g.vs.indices:
            net.add_node(i, label=labels[i], borderWidth=2, borderWidthSelected=4)
        
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
                    "max": 10
                },
                "smooth": false
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -80000,
                    "springLength": 250,
                    "springConstant": 0.001
                },
                "minVelocity": 0.75
            }
    }""")
    net.show("mygraph.html")

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
