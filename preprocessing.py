import networkx as nx
import csv
import pandas as pd
from node2vec import Node2Vec
import numpy as np
import pickle

Gph = nx.DiGraph()
with open('train_edited.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        Gph.add_node(row['user-id'])
        Gph.add_node(row['game_id'])
        if row['play_time'] == '':
            Gph.add_edge(row['user-id'], row['game_id'], weight=0)
        else:
            Gph.add_edge(row['user-id'], row['game_id'], weight=float(row['play_time']))

data = pd.read_csv('test_edited.csv')
user_counts = data['user-id'].value_counts()
all_test_id = data['user-id'].unique()

half_count = user_counts // 2
half_rows = data[data['user-id'].isin(all_test_id[:len(half_count)])].head(half_count.sum())
print(half_rows)

for index, row in half_rows.iterrows():
    Gph.add_node(row['user-id'])
    Gph.add_node(row['game_id'])
    if row['play_time'] == '':
        Gph.add_edge(row['user-id'], row['game_id'], weight=0)
    else:
        Gph.add_edge(row['user-id'], row['game_id'], weight=float(row['play_time']))

node2vec = Node2Vec(Gph, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

embeddings = model.wv

train_edges = []
train_labels = []

train_data = pd.read_csv('train_edited.csv')
train_edges = [(f"user_{row['user-id']}", f"game_{row['game_id']}") for _, row in train_data.iterrows()]
train_labels = [1] * len(train_edges)

non_edges = list(nx.non_edges(Gph))
train_edges.extend(non_edges[:len(train_edges)])
train_labels.extend([0] * len(non_edges[:len(train_edges)]))

def get_edge_features(edge):
    return np.concatenate([embeddings[edge[0]], embeddings[edge[1]]])

train_features = np.array([get_edge_features(edge) for edge in train_edges])
train_labels = np.array(train_labels)

with open('train_features', 'wb') as f:
    pickle.dump(train_features, f)

with open('train_labels', 'wb') as f:
    pickle.dump(train_labels, f)

back_half_rows = data[data['user-id'].isin(all_test_id[len(half_count):])]

test_edges = [(f"user_{row['user-id']}", f"game_{row['game_id']}") for _, row in back_half_rows.iterrows()]
test_labels = [1] * len(test_edges)

test_features = np.array([get_edge_features(edge) for edge in test_edges])
test_labels = np.array(test_labels)

with open('test_features', 'wb') as f:
    pickle.dump(test_features, f)

with open('test_labels', 'wb') as f:
    pickle.dump(test_labels, f)




