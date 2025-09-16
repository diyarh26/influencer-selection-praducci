import numpy as np
import networkx as nx
import random
import pandas as pd
import csv


NoseBook_path = 'NoseBook_friendships.csv'
cost_path = 'costs.csv'


def influencers_submission(ID1, ID2, lst):
    with open(f'{ID1}_{ID2}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lst)
def create_graph(edges_path: str) -> nx.Graph:
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net
def buy_products(net: nx.Graph, purchased: set) -> set:
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)
def product_exposure_score(net: nx.Graph, purchased_set: set) -> int:
    exposure = 0
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))

        if user in purchased_set:
            exposure += 1
        elif len(neighborhood.intersection(purchased_set)) != 0:
            b = len(neighborhood.intersection(purchased_set))
            rand = random.uniform(0, 1)
            if rand < 1 / (1 + 10 * np.exp(-b/2)):
                exposure += 1
    return exposure
def get_influencers_cost(cost_path: str, influencers: list) -> int:
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])

##OUR CODE
def cluster_graph_by_neighbors(G):
    communities = list(nx.community.greedy_modularity_communities(G))
    clusters = {frozenset(community): list(community) for community in communities}
    return clusters
def get_top_users_by_degree_and_betweenness(cluster, degrees, betweenness, costs, top_n=3):
    sorted_by_degree = [node for node in sorted(cluster, key=lambda node: degrees[node], reverse=True) if costs[costs['user'] == node]['cost'].iloc[0] == 100][:top_n]
    sorted_by_betweenness = [node for node in sorted(cluster, key=lambda node: betweenness[node], reverse=True) if costs[costs['user'] == node]['cost'].iloc[0] == 100][:top_n]
    return list(set(sorted_by_degree + sorted_by_betweenness))

def generate_combinations(top_users):
    combinations = []
    for i in range(len(top_users)):
        for j in range(i + 1, len(top_users)):
            combinations.append((top_users[i], top_users[j]))
    return combinations
def run_simulation(net: nx.Graph, influencers: list, rounds: int = 10):
    scores = []
    for _ in range(rounds):
        purchased = set(influencers)
        for i in range(6):
            purchased = buy_products(net, purchased)
        score = product_exposure_score(net, purchased)
        scores.append(score)

    avg_score = np.mean(scores)
    return avg_score
def evaluate_combinations(net, clusters, costs, degrees, betweenness, budget):
    # Get top users from the first and second clusters
    top_users_cluster_1 = get_top_users_by_degree_and_betweenness(clusters[0], degrees, betweenness, costs)
    top_users_cluster_2 = get_top_users_by_degree_and_betweenness(clusters[1], degrees, betweenness, costs)

    # Generate all possible combinations of picking 2 users from each cluster
    combinations_cluster_1 = generate_combinations(top_users_cluster_1)
    combinations_cluster_2 = generate_combinations(top_users_cluster_2)

    # Get top users from clusters 3 to 8
    top_users_other_clusters = []
    for cluster in clusters[2:8]:
        sorted_nodes = [node for node in sorted(cluster, key=lambda node: degrees[node], reverse=True) if costs[costs['user'] == node]['cost'].iloc[0] == 100]
        if sorted_nodes:
            top_users_other_clusters.append(sorted_nodes[0])

    best_group = None
    best_average_score = -1
    avg_score = -1

    # Evaluate all combinations
    for combo1 in combinations_cluster_1:
        for combo2 in combinations_cluster_2:
            combination_set = set(combo1 + combo2 + tuple(top_users_other_clusters))
            combination_cost = sum([costs[costs['user'] == user]['cost'].iloc[0] for user in combination_set if
                                    not costs[costs['user'] == user]['cost'].empty])
            if combination_cost <= budget:
                avg_score = run_simulation(net, combination_set)

                if avg_score > best_average_score:
                    best_average_score = avg_score
                    best_group = combination_set

    return best_group, best_average_score

if __name__ == '__main__':

    print("STARTING")

    NoseBook_network = create_graph(NoseBook_path)
    degrees = dict(NoseBook_network.degree())
    betweenness = nx.betweenness_centrality(NoseBook_network)
    clusters = list(cluster_graph_by_neighbors(NoseBook_network).values())
    costs = pd.read_csv(cost_path)
    budget = 1000
    influencers, temp = evaluate_combinations(NoseBook_network, clusters, costs, degrees, betweenness, budget)
    influencers_cost = get_influencers_cost(cost_path, influencers)

    print("Influencers cost: ", influencers_cost)
    if influencers_cost > 1000:
        print("***** Influencers are too expensive! *****")
        exit()

    purchased = set(influencers)

    for i in range(6):
        purchased = buy_products(NoseBook_network, purchased)
        print("finished round", i + 1)

    score = product_exposure_score(NoseBook_network, purchased)

    print("***** Your final score is " + str(score) + " *****")

    #generate a csv file of the influencers
    influencers_submission('213932338', '214034621', influencers)

    #prints the avarage of the 10 semiulations for the chosen influencers
    print(temp)


#####CODES THAT WE USED TO HELP US UNDERSTAND AND CLACULATE VALUES BUT DOSENT CONTRIBUTE TO THE FINAL CODE
 #### PRINTS THE NODES IN THE CLUSTER AND THE LENGTH OF THE CLUSTER (HOW MANY NODES IN THE CLUSTER)
# def print_clusters(clusters, data):
#     for cluster_key, nodes in clusters.items():
#         print(f"Cluster with {len(nodes)} nodes:")
#         for node in nodes:
#             print(f"Node: {node}, Cost: {data.loc[node, 'cost']}")
#         print("\n")
