import pandas as pd
import pickle

movies = pd.read_csv('movies.dat', sep='::', names=['id', 'title', 'genres'],
                     engine='python', encoding='latin-1')
movies.set_index('id', inplace=True)


with open('item_mapping.pkl', 'rb') as f:
    id2item = pickle.load(f)

for sasrec_idx in range(1, 6):
    real_id = id2item[sasrec_idx]
    try:
        title = movies.loc[real_id]['title']
        print(f"Index SASRec {sasrec_idx} -> Real ID {real_id} -> Film: {title}")
    except KeyError:
        print(f"Index SASRec {sasrec_idx} -> Real ID {real_id} -> (Movie not found in movies.dat)")

dataset = pd.read_csv('ml-1m.txt', sep=' ', names=['user', 'item'], header=None,
                     engine='python')

