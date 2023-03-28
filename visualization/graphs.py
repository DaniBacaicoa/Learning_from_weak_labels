def plot_label_graph(samples):
    G = nx.Graph()

    # Add nodes to graph
    for i, sample in enumerate(samples):
        G.add_node(i, pos=sample[0], label=set(sample[1]))

    # Add edges between nodes with shared labels
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples[i + 1:], i + 1):
            if len(sample_i[1].intersection(sample_j[1])) > 0:
                G.add_edge(i, j)

    # Draw graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, node_color='lightblue', with_labels=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
    plt.show()



def plot_prediction_graph(samples, y_pred):
    G = nx.Graph()

    # Add nodes to graph
    for i, sample in enumerate(samples):
        G.add_node(i, pos=sample[0], label=y_pred[i])

    # Draw graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    node_colors = [y_pred[i] for i in range(len(samples))]
    nx.draw(G, pos, node_color=node_colors, with_labels=True, cmap='tab10')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
    plt.show()

