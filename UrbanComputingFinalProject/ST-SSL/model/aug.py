import os
import copy
import numpy as np 
import torch 
import networkx as nx
from scipy.stats import kstest, powerlaw
import random

import pandas as pd

def rewire_to_small_world(G,sim, p, only_distance = False):
    n = len(G.nodes())
    
    degree_sequence = [d for n, d in G.degree()]
    
    if not only_distance:
    # Ensure the matrix is symmetric


        # Iterate through each edge
        for i in range(n):
            for j in range(i + 1, n):
                # Check if there is an edge
                if G.has_edge(i,j):
                    # Rewire the edge with probability p
                    if np.random.rand() < p:
                        
                        # Choose a random node (excluding the source node and its neighbors)
                        choices = np.delete(np.arange(n), [i, *np.where(nx.to_numpy_array(G)[i, :] == 1)[0]])
                        new_j = np.random.choice(choices)

                        # Rewire the edge
                        if G.degree(new_j < np.mean(degree_sequence)) & G.degree(j > np.mean(degree_sequence)):
                            G.remove_edge(i,j)
                            G.add_edge(i, new_j)

    else:
        probability_of_edge = p 

# Randomly add edges based on the specified probability
        for node1 in range(n):
            for node2 in range(node1 + 1, n):
                if (random.random() < probability_of_edge) & (sim[node1,node2] < 0.9) & G.has_edge(node1,node2) & G.degree(node1) != np.min(G.degree()) & G.degree(node2)!= np.min(G.degree()):
            # Set both entries to 1 to represent an undirected graph
                   G.remove_edge(node1, node2)
    
    return G

# Example usage
n = 20  # Number of nodes
p = 0.3  # Rewiring probability

def generate_power_law_graph(G, similarity_matrix, alpha= 0.01):
    # Create a graph from the adjacency matrix
    

    # Get the number of nodes in the graph
    num_nodes = G.number_of_nodes()

    # Add edges based on power-law distribution and node similarities
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (((G.degree(j)))/np.sum(list(dict(G.degree()).values()))>alpha) :
                G.add_edge(i, j)

    return G

def sparsity_correction(G, p, sim_mx):
    # Get the list of edges and corresponding weights based on the edge power matrix
    sim_mx = sim_mx.numpy()
    edges = list(G.edges())
    weights = sim_mx.flatten()

    # Normalize weights to create probabilities
    probabilities = np.array(weights) / np.sum(weights)

    
    # Randomly select edges to delete based on the specified probability (p)
    edges_to_delete = np.random.choice(len(edges), size=int(p * len(edges)), replace=False, p=probabilities)

    # Delete the selected edges from the graph
    edges_to_delete = [edges[i] for i in edges_to_delete]
    G.remove_edges_from(edges_to_delete)
    

    return G



def small_world_test(input_graph, output_folder="aug_graphs_smallworld"):
    
    #----------------------------------------------------------
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # # use a counter to keep track of the number of calls of this function

    # small_world_test_counter = small_world_test.small_world_test_counter if hasattr(small_world_test, 'small_world_test_counter') else 0

    #----------------------------------------------------------

    gamma = 0.57722
    
    adj_matrix =input_graph.numpy()
    G = nx.from_numpy_array(adj_matrix)

    all_pair_distances = nx.floyd_warshall(G)
    mean_distance = sum([dist for node_distances in all_pair_distances.values() for dist in node_distances.values()]) / (len(G) * (len(G) - 1))

    
    average_clustering_coefficient = nx.average_clustering(G)

    observed_edges = len(G.edges())
    possible_edges = len(G.nodes()) * (len(G.nodes()) - 1) / 2  # For an undirected graph
    p = observed_edges / possible_edges

    kappa = len(G.nodes())* p

    ER_clustering_coefficient = p
    ER_mean_distance = np.log(len(G.nodes()) - gamma)/np.log(kappa) + 0.5

    # ------------------------------------------------------------
    result = (abs(mean_distance/ER_mean_distance) < 2) & (abs(average_clustering_coefficient/ER_clustering_coefficient) > 1)
    # filename = os.path.join(output_folder, f"small_world_test_{small_world_test_counter}")
    # pd.DataFrame([result], columns=["result"]).to_csv(filename)

    # small_world_test_counter += 1

    # ------------------------------------------------------------

    return  (abs(mean_distance/ER_mean_distance) < 2) & (abs(average_clustering_coefficient/ER_clustering_coefficient) > 1)
    

def sparsity_test(input_graph, output_folder="aug_graphs_sparsity"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # use a counter to keep track of the number of calls of this function
    aug_counter = aug_topology.aug_counter if hasattr(aug_topology, 'aug_counter') else 0

    adj_matrix = input_graph.numpy()
    G = nx.from_numpy_array(adj_matrix)

    observed_edges = len(G.edges())
    possible_edges = len(G.nodes()) * (len(G.nodes()) - 1) / 2  # For an undirected graph
    p = observed_edges / possible_edges
    return p > 0.2

def power_law_test(input_graph, output_folder="aug_graphs_powerlow"):
    
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # use a counter to keep track of the number of calls of this function
    aug_counter = aug_topology.aug_counter if hasattr(aug_topology, 'aug_counter') else 0


    adj_matrix = input_graph.numpy()
    G = nx.from_numpy_array(adj_matrix)

    degree_sequence = [d for n, d in G.degree()]
    

    fit_alpha, fit_loc, fit_beta = powerlaw.fit(degree_sequence)
    ks_statistic, ks_p_value = kstest(degree_sequence, 'powerlaw', args=(fit_alpha, fit_loc, fit_beta))

    return(ks_p_value >= 0.05)

def sim_global(flow_data, sim_type='cos'):
    """Calculate the global similarity of traffic flow data.
    :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n,l,v,c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n,v,c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5 
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')
    
    return sim

def aug_topology(sim_mx, input_graph, percent=0.2, output_folder="aug_graphs_smallworld"):
    """Generate the data augumentation from topology (graph structure) perspective 
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """    
    
 
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # use a counter to keep track of the number of calls of this function

    small_world_test_counter = aug_topology.small_world_test_counter if hasattr(aug_topology, 'small_world_test_counter') else 0

    #----------------------------------------------------------
    
    ## edge dropping starts here
    drop_percent = percent / 2
    
    index_list = input_graph.nonzero() # list of edges [row_idx, col_idx]
    
    edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = int(edge_num * drop_percent / 2) 
    aug_graph = copy.deepcopy(input_graph) 

    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).numpy() # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    drop_index = index_list[drop_list]
    
    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    ## edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)] # .numpy()
    add_prob = torch.softmax(add_prob, dim=0).numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2), 
                                size=add_drop_num, p=add_prob)
    
    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones



    result = small_world_test(aug_graph)
    result2 = sparsity_test(aug_graph)
    result3 = power_law_test(aug_graph)
    
    # filename = os.path.join(output_folder, "small_world_test_0.txt")

    # filename = output_folder + "/small_world_test_0.txt"


    # with open(filename, 'a') as result_file:
    #     result_file.write(str(int(result)))
    #     result_file.write(',')
    #     result_file.write(str(int(result2)))
    #     result_file.write(',')
    #     result_file.write(str(int(result3)) + '\n')

        

    # small_world_test_counter += 1


    
   
    G = aug_graph.numpy()
    G = nx.from_numpy_array(G)

    observed_edges = len(G.edges())
    possible_edges = len(G.nodes()) * (len(G.nodes()) - 1) / 2  # For an undirected graph
    p = observed_edges / possible_edges

    if result3:
        G = generate_power_law_graph(G, sim_mx, 0.009)
    if result:
        G = rewire_to_small_world(G, sim_mx, 0.05, only_distance= False)
        G = rewire_to_small_world(G, sim_mx, 0.7, only_distance= True)
    if result2:
        G = sparsity_correction(G, p - 0.2, sim_mx)
    G = nx.to_numpy_array(G)
    G = torch.from_numpy(G)
    G = G.type(dtype = torch.float32)

    return G

def aug_traffic(t_sim_mx, flow_data, percent=0.2):
    """Generate the data augumentation from traffic (node attribute) perspective.
    :param t_sim_mx: temporal similarity matrix after softmax, [l,n,v]
    :param flow_data: input flow data, [n,l,v,c]
    """
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = copy.deepcopy(flow_data)

    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).numpy()
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)

    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list], 
        y.reshape(-1)[mask_list], 
        z.reshape(-1)[mask_list]] = zeros 
    


    return aug_flow
