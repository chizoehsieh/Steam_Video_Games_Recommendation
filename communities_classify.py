import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import random
from netgraph import Graph
import gravis as gv


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


############ preprocessing  ###############
Gph = nx.DiGraph()

train_data = pd.read_csv('train_edited.csv')
for index, row in train_data.iterrows():
    Gph.add_node(row['user-id'])
    Gph.add_node(row['game_id'])
    if row['play_time'] == '':
        Gph.add_edge(row['user-id'], row['game_id'], weight=0)
    else:
        Gph.add_edge(row['user-id'], row['game_id'], weight=float(row['play_time']))

data = pd.read_csv('test_edited.csv')

for index, row in data.iterrows():
    Gph.add_node(row['user-id'])
    Gph.add_node(row['game_id'])
    if row['play_time'] == '':
        Gph.add_edge(row['user-id'], row['game_id'], weight=0)
    else:
        Gph.add_edge(row['user-id'], row['game_id'], weight=float(row['play_time']))

print('after drawing')

############## 社群分析 ################
Gph_undirec = Gph.to_undirected()
subgraph = Gph.subgraph([node for node in Gph.nodes() if int(node) >= 5250])
subgraph_undirec = subgraph.to_undirected()

communities = greedy_modularity_communities(Gph_undirec)

# user_partition = {node: community for node, community in communities.items() if int(node) >= 5250}

community_dict = {}
for i, community in enumerate(communities):
    for node in community:
        if int(node) >= 5250:
            community_dict[node] = i
        else:
            community_dict[node] = len(communities) + 1

# nx.set_node_attributes(Gph, community_dict, 'community')

unique_communities = list(set(community_dict.values()))

print('community count: ', len(communities))
# print(community_dict)

colors = list(mcolors.CSS4_COLORS.values())
colors[len(communities) + 1] = '#FFFFFF'
# print(colors)
for node in Gph.nodes():
    Gph.nodes[node]['color'] = colors[community_dict[node]]
# random.shuffle(colors)
# color_map = {community: colors[i % len(colors)] for i, community in enumerate(unique_communities)}

# node_colors = [color_map[community_dict[node]] for node in Gph.nodes() if node in community_dict]

print(f"Total nodes: {len(Gph.nodes())}")
print(f"Total edges: {len(Gph.edges())}")
# print(f"Node colors assigned: {len(node_colors)}")


# subgraph = Gph.subgraph(list(Gph.nodes())[:100])



# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(subgraph, seed=42)
# nx.draw_spring(subgraph, node_color=[node_colors[i] for i in range(100)], node_size=25)
# # nx.draw_networkx_edges(subgraph, pos, width=0.2, edge_color='gray')
# plt.title("User Communities (Sample Subgraph)")
# plt.show()

# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(Gph, seed=42)
# nx.draw_networkx_nodes(Gph, pos, node_color = node_colors, node_size = 25)
# nx.draw_networkx_edges(Gph_undirec, pos, width=0.2, edge_color='gray', edge_cmap='viridis')
# # nx.draw(Gph, pos, node_color = node_colors, with_labels = False, node_size = 25, edge_color = 'gray')
# plt.title("User Communities")
# plt.show()

# values = [community_dict[node] for node in list(Gph_undirec.nodes())[:100]]
# color_map = {community: colors[i % len(colors)] for i, community in enumerate(unique_communities)}
# node_color = {node: color_map[community_dict[node]] for node in Gph_undirec.nodes()}
# nx.draw_spring(subgraph, cmap = plt.get_cmap('viridis'), node_color = node_color, node_size = 25, with_labels = False)
# Graph(Gph, node_color = node_color, node_size = 25, with_labels = False)
# plt.show()

pos = community_layout(Gph, community_dict)
nx.draw(Gph, pos, cmap = plt.get_cmap('viridis'), node_color = list(community_dict.values()), with_labels = False, node_size = 25, edge_color = 'gray')
plt.show()
# fig = gv.d3(Gph, use_node_size_normalization=True, node_size_normalization_max=30,
#       use_edge_size_normalization=True, edge_size_data_source='weight', edge_curvature=0.3)
# fig.export_png('communities_graph.png')