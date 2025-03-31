import pandas as pd

df = pd.read_csv('社群媒體期末專案_資料/combine_dataset.csv')
game_titles = df['game-title'].unique()
game_titles_id = {title: i for i, title in enumerate(game_titles)}


df = pd.read_csv('社群媒體期末專案_資料/train_0607.csv')
# game_titles = df['game-title'].unique()
print(game_titles)

# game_titles_id = {title: i for i, title in enumerate(game_titles)}
df['game_id'] = df['game-title'].map(game_titles_id)

df.to_csv('train_edited_0607.csv', index=False)

df = pd.read_csv('社群媒體期末專案_資料/test_0607.csv')
df['game_id'] = df['game-title'].map(game_titles_id).astype(int)
df.to_csv('test_edited_0607.csv', index=False)