import networkx as nx
import matplotlib.pyplot as plt
import torch

class NNVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_weights(self):
        weights = []
        for param in self.model.parameters():
            if len(param.shape) == 2:
                weights.append(param.detach().numpy())

        G = nx.DiGraph()
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    G.add_edge('Layer {} Node {}'.format(i, j),
                               'Layer {} Node {}'.format(i+1, k),
                               weight=weights[i][j][k])

        pos = nx.kamada_kawai_layout(G)
        edge_labels = {(u, v): '{:.2f}'.format(d['weight']) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def plot_weights_and_values(self, x):
        weights = []
        for param in self.model.parameters():
            if len(param.shape) == 2:
                weights.append(param.detach().numpy())

        G = nx.DiGraph()
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    G.add_edge('Layer {} Node {}'.format(i, j),
                               'Layer {} Node {}'.format(i+1, k),
                               weight=weights[i][j][k])

        pos = nx.kamada_kawai_layout(G)
        edge_labels = {(u, v): '{:.2f}'.format(d['weight']) for u, v, d in G.edges(data=True)}
        node_sizes = [abs(x[i])*100 for i in range(len(x))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()




def visualize_weights(net):
    G = nx.DiGraph()

    # Add nodes to graph
    for i, (name, param) in enumerate(net.named_parameters()):
        G.add_node(name, size=torch.abs(param).sum().item())

    # Add edges to graph
    for name1, param1 in net.named_parameters():
        for name2, param2 in net.named_parameters():
            if name1 != name2:
                G.add_edge(name1, name2, weight=torch.abs(param1 * param2).sum().item())

    # Draw graph
    pos = nx.circular_layout(G)
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    edge_weights = [G.edges[e]['weight'] for e in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=edge_weights)
    nx.draw_networkx_labels(G, pos)
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt
import torch

def visualize_activation(net, x):
    G = nx.DiGraph()

    # Add nodes to graph
    for i, (name, param) in enumerate(net.named_parameters()):
        G.add_node(name, size=0)

    # Add edges to graph
    for name1, param1 in net.named_parameters():
        for name2, param2 in net.named_parameters():
            if name1 != name2:
                G.add_edge(name1, name2, weight=torch.abs(param1 * param2).sum().item())

    # Compute activations
    activations = net(x)
    for i, a in enumerate(activations):
        G.nodes[f'layer_{i}_out']['size'] = torch.abs(a).sum().item()

    # Draw graph
    pos = nx.circular_layout(G)
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    edge_weights = [G.edges[e]['weight'] for e in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=edge_weights)
    nx.draw_networkx_labels(G, pos)
    plt.show()
