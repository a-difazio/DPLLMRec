import pandas as pd
import pickle
import os

data_dir = 'ml-1m'

dataset = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::', names=['user', 'item', 'rating', 'timestamp'],
                      engine='python')
processed = (dataset.drop(['rating'], axis=1).sort_values(by=['user', 'timestamp'])
             .reset_index(drop=True).drop('timestamp', axis=1))

unique_users = processed['user'].unique()
unique_items = processed['item'].unique()

user2id = {old: new + 1 for new, old in enumerate(unique_users)}
item2id = {old: new + 1 for new, old in enumerate(unique_items)}

# inverted maps
id2item = {v: k for k, v in item2id.items()}
id2user = {v: k for k, v in user2id.items()}

with open(os.path.join(data_dir, 'item_mapping.pkl'), 'wb') as f:
    pickle.dump(id2item, f)

with open(os.path.join(data_dir, 'user_mapping.pkl'), 'wb') as f:
    pickle.dump(id2user, f)

processed['user'] = processed['user'].map(user2id)
processed['item'] = processed['item'].map(item2id)

# sanity check
print(f"Unique items: {processed['item'].nunique()}")
print(f"Max item id: {processed['item'].max()}, Min item id: {processed['item'].min()}")

print(f"Unique users: {processed['user'].nunique()}")
print(f"Max user id: {processed['user'].max()}, Min user id: {processed['user'].min()}")

processed = processed.astype(int)
processed.to_csv(os.path.join(data_dir, 'ml-1m.txt'), sep=' ', index=False, header=False)
