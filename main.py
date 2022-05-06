import numpy as np
import networkx as nx
import random
import pandas as pd


def simulate_edge_creation(G, prob):
    new_G = nx.create_empty_copy(G, with_data=True)
    nodes1 = list(G.nodes)
    for i in range(len(nodes1)):
        for j in range(i, len(nodes1)):
            n1 = nodes1[i]
            n2 = nodes1[j]
            if n1 != n2 and G.has_edge(n1, n2) == False:
                common_friends = len(sorted(nx.common_neighbors(G, n1, n2)))
                rand_num = np.random.rand()
                if rand_num < prob[common_friends]:
                    new_G.add_edge(n1, n2)

    G.add_edges_from(new_G.edges)


def infection(G):
    infected_list = []
    nodes1 = list(G.nodes)
    for node in nodes1:
        if G.nodes[node]['bought'] != 1:
            n_list = list(G.neighbors(node))
            nt = len(n_list)
            bt = 0
            for neighbor in n_list:
                if G.nodes[neighbor]['bought'] == 1:
                    bt += 1
            h = G.nodes[node]['h']

            if h == 0:
                infection_prob = bt / nt
            else:
                infection_prob = (h * bt) / (1000 * nt)

            u = np.random.rand()

            if u <= infection_prob:
                infected_list.append(node)

        for node in infected_list:
            G.nodes[node]['bought'] = 1

## יש בעיה כי חלק מהצמתים לא מכירים את הפרמטר H

def FINISH_HIM(G, artist_num, prob):
    spot = pd.read_csv('spotifly.csv', index_col=False)
    new_spot = spot[spot[' artistID'] == artist_num]
    new_spot = new_spot.reset_index(drop=True)
    values = new_spot.values.tolist()
    nx.set_node_attributes(G, 0, "h")
    nx.set_node_attributes(G, 0, "bought")
    nx.set_node_attributes(G, 0, "grade")

    dic1 = {}
    for value in values:
        dic1[value[0]] = {'h': value[2]}
    nx.set_node_attributes(G, dic1)
    influencers_list = get_top_influencers(G)
    infect_nodes(G, influencers_list)
    acc = simulate_graph(G, prob)
    return acc


def infect_nodes(G, list_of_nodes):
    for node in list_of_nodes:
        G.nodes[node]['bought'] = 1


def artist_grade(G, node1, listen_to_artist):
    count = 0
    for listener in listen_to_artist:
        if listener != node1:
            d = nx.shortest_path_length(G, source=node1, target=listener)
            h_grade = G.nodes[listener]['h'] - 1000
            count += (1 / d) * h_grade

    return count / len(listen_to_artist)


def create_grade(G):
    nodes1 = list(G.nodes)
    L = []
    for node in nodes1:
        if G.nodes[node]['h'] != 0:
            L.append(node)
    artists = []
    deg = []
    for node in nodes1:  # for normalization purposes
        artists.append(artist_grade(G, node, L))
        deg.append(G.degree[node])

    max_deg = max(deg)
    max_artist = max(artists)
    min_artists = -999

    for node in nodes1:
        a = artist_grade(G, node, L)
        if a >= 0:
            a = a / max_artist
        else:
            a = a / min_artists
        c = nx.closeness_centrality(G, node)
        d = G.degree[node] / max_deg
        x = (0.2 * a + 0.4 * c + 0.4 * d)
        #         x = (a +c+d)/3

        G.nodes[node]['grade'] = x


def get_top_influencers(G):
    create_grade(G)
    dic = nx.get_node_attributes(G, 'grade')
    sorted_dic = sorted(dic, key=dic.get, reverse=True)
    return sorted_dic[:5]


def simulate_graph(G, prob):
    create_grade(G)
    for i in range(5):
        simulate_edge_creation(G, prob)  ## from t= 0 to t=1
        infection(G)
    nodes1 = list(G.nodes)
    count = 0
    for node in nodes1:
        if G.nodes[node]['bought'] == 1:
            count += 1
    return count / len(nodes1)


df = pd.read_csv('instaglam_1.csv')
G = nx.from_pandas_edgelist(df, source="userID", target="friendID")

df0 = pd.read_csv('instaglam0.csv')
G0 = nx.from_pandas_edgelist(df0, source="userID", target="friendID")

S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
S0 = [G0.subgraph(c).copy() for c in nx.connected_components(G0)]

nodes1 = list(S[0].nodes)
list_of_prob = []
histogram = [0] * 1000
every_group = [[0]] * 1000
for i in range(len(nodes1)):
    for j in range(i, len(nodes1)):
        n1 = nodes1[i]
        n2 = nodes1[j]
        if n1 != n2 and (n2 not in S[0].neighbors(n1)):
            num = len(sorted(nx.common_neighbors(S[0], n1, n2)))
            if num != 0:
                histogram[num] += 1
                if every_group[num] == [0]:
                    every_group[num] = [[n1, n2]]
                else:
                    every_group[num].append([n1, n2])

numerator = [0] * 1000
L = list(S0[0].edges())

for i in range(1, 1000):  ## possible to use 50 instead of 1000
    for edge in every_group[i]:
        if type(edge) != int:
            temp = (edge[0], edge[1])
            if G0.has_edge(temp[0], temp[1]):
                numerator[i] += 1

prob = [0] * 1000
for i in range(0, 1000):
    if histogram[i] != 0:
        prob[i] = numerator[i] / histogram[i]

p = prob[:23]
x = np.mean(p)
for i in range(3, 1000):
    if prob[i] == 0.0:
        prob[i] = x

c = FINISH_HIM(S0[0],150,prob)
print(c)