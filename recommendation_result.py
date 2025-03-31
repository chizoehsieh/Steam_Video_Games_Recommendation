import pandas as pd
import numpy as np
import csv
import ast
from matplotlib import pyplot as plt


recommendation_data = pd.read_csv('recommendations.csv')
purchased_data = pd.read_csv('purchased.csv')


purchased = dict()
recommen = dict()
percission = dict()

for i in range(len(recommendation_data)):
    purchased[purchased_data['user-id'][i]] = ast.literal_eval(purchased_data['purchased games'][i])
    recommen[recommendation_data['user-id'][i]] = [item[0] for item in ast.literal_eval(recommendation_data['recomment game'][i])]

user_ids = list(purchased.keys())
print(user_ids)
for i in range(len(user_ids)):
    count = 0
    user_id = user_ids[i]
    for game in purchased[user_id]:
        if game in recommen[user_id]:
            count += 1

    percission[user_id] = count / len(purchased[user_id])


accuracy_values = [percission[user_id] for user_id in user_ids]

# plt.plot(user_ids, accuracy_values)
# plt.xlabel('User ID')
# plt.ylabel('Accuracy')
# plt.title('User Accuracy')
# plt.show()
print(np.mean(accuracy_values))
x = list(range(len(user_ids)))
plt.bar(x, accuracy_values)
plt.xlabel('User ID')
plt.ylabel('Accuracy')
plt.title('Recommendation Accuracy')
plt.show()