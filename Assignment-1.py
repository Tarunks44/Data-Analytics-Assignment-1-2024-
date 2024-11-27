import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from functools import lru_cache
import itertools
import time
import pandas as pd

# Function to reindex the graph nodes
def reindex_graph(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G, mapping

# Load the Wiki-Vote dataset
def load_wiki_vote():
    wiki_vote_edges = []
    with open('Wiki-Vote.txt', 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            node1, node2 = map(int, line.split())
            wiki_vote_edges.append((node1, node2))

    G_wiki = nx.Graph()
    G_wiki.add_edges_from(wiki_vote_edges)
    return G_wiki

# Load the Last.fm Asia dataset
def load_lastfm_asia():
    lastfm_edges_df = pd.read_csv('lastfm_asia_edges.csv')

    G_lastfm = nx.Graph()
    G_lastfm.add_edges_from(lastfm_edges_df.values)
    return G_lastfm

@lru_cache(maxsize=None)
def compute_shortest_paths(G):
    return dict(nx.all_pairs_shortest_path_length(G))

def process_graph(G, title):
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest_cc = G.subgraph(largest_cc).copy()
    
    distance_dict = compute_shortest_paths(G_largest_cc)

    nodes = list(G_largest_cc.nodes())
    num_nodes = len(nodes)
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node2 in distance_dict[node1]:
                distance_matrix[i, j] = distance_dict[node1][node2]

    condensed_distance_matrix = squareform(distance_matrix)

    Z = linkage(condensed_distance_matrix, method='average')

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=[str(node) for node in nodes])
    plt.title(f'Dendrogram After Girvan-Newman - {title}')
    plt.xlabel('Node Index')
    plt.ylabel('Distance')
    plt.show()

    return G_largest_cc

def automated_girvan_newman_limited(G, max_splits=3):
    best_modularity = -1
    best_communities = None
    
    communities_generator = nx.algorithms.community.girvan_newman(G)
    
    for i, communities in enumerate(communities_generator):
        current_modularity = nx.algorithms.community.modularity(G, communities)
        
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = communities
        
        if i >= max_splits:
            break  # Stop after a limited number of splits
    
    return best_communities, best_modularity

def louvain_one_iter(G):
    nodes_connectivity_list = np.array(G.edges())
    
    communities = np.zeros((len(G.nodes()), 2))
    communities[:, 0] = list(G.nodes())
    communities[:, 1] = list(G.nodes())

    n_communities = np.unique(communities[:, 1])
    m = G.number_of_edges()
    
    for i in range(len(n_communities)):
        delta_mod_list = []
        cout = []
        cin = []
        for j in range(i + 1, len(n_communities)):
            c1 = communities[:, 0][communities[:, 1] == n_communities[i]]
            c2 = communities[:, 0][communities[:, 1] == n_communities[j]]
            for c in zip(c1):
                for d in zip(c2):
                    if G.has_edge(c[0], d[0]):
                        cout.append(c[0])
                        cin.append(d[0])
                        g1 = G.subgraph(c1).copy()
                        g2 = G.subgraph(c2).copy()
                        new_comm = np.concatenate((c2, c[0][np.newaxis]))
                        g3 = G.subgraph(new_comm).copy()
                        i_in = g1.number_of_edges()
                        i_tot = sum(dict(G.degree(c2)).values())
                        k_in = g3.degree(c[0])
                        k_tot = G.degree(c[0])
                        j_in = g2.number_of_edges()
                        j_tot = sum(dict(G.degree(c1)).values())
                        k_out = g1.degree(c[0])
                        mod_merge = (j_in+k_in)/2*m - (j_tot+k_tot)**2/(2*m)**2 - \
                            (j_in/2*m - (j_tot/(2*m))**2 - (k_in/2*m)**2)
                        mod_split = ((i_in-k_out)/2*m - (i_tot-k_tot)**2/(2*m)**2 - (k_out/2*m)**2) - \
                            (i_in/2*m - (i_tot/(2*m)) ** 2)
                        delta_mod = mod_merge + mod_split
                        delta_mod_list.append(delta_mod)
        if delta_mod_list == []:
            continue
        delta_mod_max_idx = np.argmax(delta_mod_list)
        communities[int(cin[delta_mod_max_idx]), 1] = min(
            communities[int(cout[delta_mod_max_idx]), 1], communities[int(cin[delta_mod_max_idx]), 1])

    G.remove_nodes_from(list(nx.isolates(G)))
    del_ind = []
    for i in range(len(communities)):
        if communities[i, 0] in G.nodes():
            continue
        else:
            del_ind.append(i)
    communities = np.delete(
        communities, np.array(del_ind, dtype=int), axis=0)
    
    return communities

def get_communities_from_partition(communities):
    community_dict = {}
    for node, community in communities:
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(int(node))
    return list(community_dict.values())

def plot_communities(G, communities, title):
    colors = itertools.cycle(plt.cm.tab20.colors)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_size=50, node_color=[next(colors)], label=f'Community {i+1}')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.show()

def compare_algorithms_limited(G, title, max_splits=3):
    start_time = time.time()
    best_communities, best_modularity = automated_girvan_newman_limited(G, max_splits=max_splits)
    girvan_newman_time = time.time() - start_time

    start_time = time.time()
    louvain_partition = louvain_one_iter(G)
    louvain_communities = get_communities_from_partition(louvain_partition)
    louvain_time = time.time() - start_time

    print(f"Girvan-Newman time ({title}): {girvan_newman_time:.2f} seconds")
    print(f"One Iteration Louvain time ({title}): {louvain_time:.2f} seconds")

    # Display Dendrogram after Girvan-Newman
    process_graph(G, title)

    # Plot Girvan-Newman communities
    plot_communities(G, best_communities, f'Communities Detected After Girvan-Newman - {title}')

    # Plot Louvain communities
    plot_communities(G, louvain_communities, f'Communities Detected After One Iteration Louvain - {title}')

    return best_communities, best_modularity, louvain_communities

# Process both datasets and analyze results

print("Processing Wiki-Vote Dataset with Limited Girvan-Newman:")
G_wiki = load_wiki_vote()
G_wiki, mapping_wiki = reindex_graph(G_wiki)
best_communities_wiki, best_modularity_wiki, louvain_communities_wiki = compare_algorithms_limited(G_wiki, "Wiki-Vote Dataset", max_splits=3)

print("\nProcessing Last.fm Asia Dataset with Limited Girvan-Newman:")
G_lastfm = load_lastfm_asia()
G_lastfm, mapping_lastfm = reindex_graph(G_lastfm)
best_communities_lastfm, best_modularity_lastfm, louvain_communities_lastfm = compare_algorithms_limited(G_lastfm, "Last.fm Asia Dataset", max_splits=3)

# Final Output Summary
# Final Output Summary
print("\nSummary of Results:")

print(f"Wiki-Vote Dataset - Best Modularity (Girvan-Newman): {best_modularity_wiki}")
print(f"Number of communities detected (Girvan-Newman): {len(best_communities_wiki)}")
print(f"Number of communities detected (One Iteration Louvain): {len(louvain_communities_wiki)}")

print(f"Last.fm Asia Dataset - Best Modularity (Girvan-Newman): {best_modularity_lastfm}")
print(f"Number of communities detected (Girvan-Newman): {len(best_communities_lastfm)}")
print(f"Number of communities detected (One Iteration Louvain): {len(louvain_communities_lastfm)}")

