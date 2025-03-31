import numpy as np
import pickle

def precision(recommended, purchased):
    relevant_and_recommended = [1 if game in purchased else 0 for game in recommended]
    return sum(relevant_and_recommended) / len(recommended)

def recall(recommended, purchased):
    relevant_and_recommended = [1 if game in recommended else 0 for game in purchased]
    return sum(relevant_and_recommended) / len(purchased)

def dcg(recommended, purchased):
    for i,game in enumerate(recommended):
        if game in purchased:
            dcg += 1 / np.log2(i+2)

    return dcg

def ndcg(recommended, purchased):
    dcg_max = sum([1 / np.log2(i + 2) for i in range(min(len(purchased), 10))])
    if not dcg_max:
        return 0.0
    return dcg(recommended, purchased) / dcg_max

def hit_rate(recommended, purchased):
    return int(any(game in purchased for game in recommended))



with open('test_recommendations', 'rb') as f:
    recommendateon_data = pickle.load(f)

with open('back_half_rows', 'rb') as f:
    purchased_data = pickle.load(f)

print('recommendat: ', recommendateon_data)
print('purchased: ', purchased_data)

all_test_id = purchased_data['user-id'].unique()

purchased = dict()

for user_id in all_test_id:
    purchased[user_id] = []

for index, row in purchased_data.iterrows():
    purchased[row['user-id']].append(row['game_id'])

precisions = []
recalls = []
ndcgs = []
hit_rates = []

for user_id in all_test_id:
    precisions.append(precision(recommendateon_data[user_id], purchased[user_id]))
    recalls.append(recall(recommendateon_data[user_id], purchased[user_id]))
    ndcgs.append(ndcg(recommendateon_data[user_id], purchased[user_id]))
    hit_rates.append(hit_rate(recommendateon_data[user_id], purchased[user_id]))

