import pandas as pd
import pickle
import os
import argparse
import json

DATA_DIR = 'raw'
RESULTS_DIR = 'preprocessed'

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument( '--threshold', type=float)
    parser.add_argument( '--kcore', type=int)
    parser.add_argument( '--split', type=float)
    return vars(parser.parse_args())

def load_movielens():
    return pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::',
                          names=['user', 'item', 'rating', 'timestamp'],
                          engine='python')

def load_amazon_beauty():
    return pd.read_csv(os.path.join(DATA_DIR, 'All_Beauty.csv'), sep=',',
                          names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_toys():
    return pd.read_csv(os.path.join(DATA_DIR, 'Toys_and_Games.csv'), sep=',',
                          names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_books():
    return pd.read_csv(os.path.join(DATA_DIR, 'Books.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_music():
    return pd.read_csv(os.path.join(DATA_DIR, 'Digital_Music.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_cds():
    return pd.read_csv(os.path.join(DATA_DIR, 'CDs_and_Vinyl.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_movies():
    return pd.read_csv(os.path.join(DATA_DIR, 'Movies_and_TV.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_sports():
    return pd.read_csv(os.path.join(DATA_DIR, 'Sports_and_Outdoors.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_fashion():
    return pd.read_csv(os.path.join(DATA_DIR, 'AMAZON_FASHION.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_clothing():
    return pd.read_csv(os.path.join(DATA_DIR, 'Clothing_Shoes_and_Jewelry.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])

def load_amazon_games():
    return pd.read_csv(os.path.join(DATA_DIR, 'Video_Games.csv'), sep=',',
                       names=['user', 'item', 'rating', 'timestamp'])


def binarize(dataset, threshold):
    if threshold > 0:
        dataset = dataset[dataset.rating >= threshold]
    return dataset.drop(columns=['rating'])

def kcore_filter(dataset, kcore):
    while True:
        before = len(dataset)

        user_counts = dataset['user'].value_counts()
        item_counts = dataset['item'].value_counts()

        dataset = dataset[
            dataset['user'].isin(user_counts[user_counts >= kcore].index)
        ]
        dataset = dataset[
            dataset['item'].isin(item_counts[item_counts >= kcore].index)
        ]

        if len(dataset) == before:
            break

    # sanity check
    user_counts = dataset.groupby('user')['item'].count()
    item_counts = dataset.groupby('item')['user'].count()

    print(f"Min interactions per user: {user_counts.min()}")
    print(f"Min interactions per item: {item_counts.min()}")

    return dataset.reset_index(drop=True)

def encode_ids(dataset):
    dataset = dataset.sort_values(by=['user', 'timestamp']).reset_index(drop=True)

    unique_users = dataset['user'].unique()
    unique_items = dataset['item'].unique()

    user2id = {old: new + 1 for new, old in enumerate(unique_users)}
    item2id = {old: new + 1 for new, old in enumerate(unique_items)}

    id2item = {v: k for k, v in item2id.items()}
    id2user = {v: k for k, v in user2id.items()}

    dataset['user'] = dataset['user'].map(user2id)
    dataset['item'] = dataset['item'].map(item2id)

    # sanity check
    print(f"Min user id: {dataset['user'].min()}, "
          f"Max user id: {dataset['user'].max()}")

    print(f"Min item id: {dataset['item'].min()}, "
          f"Max item id: {dataset['item'].max()}")

    return dataset, id2user, id2item

def split(dataset, fraction):
    if fraction == 0.0:
        return dataset, None

    train = []
    test = []

    for user, group in dataset.groupby('user'):
        group = group.sort_values(by=['timestamp'])

        n_test = min(len(group) - 1, max(1, int(len(group) * fraction)))

        train.append(group.iloc[:-n_test])
        test.append(group.iloc[-n_test:])

    train = pd.concat(train).reset_index(drop=True)
    test = pd.concat(test).reset_index(drop=True)

    return train.drop('timestamp', axis=1), test.drop('timestamp', axis=1)

def print_stats(dataset):
    seq_lengths = dataset.groupby('user')['item'].count()

    n_users = dataset['user'].nunique()
    n_items = dataset['item'].nunique()
    n_interactions = len(dataset)
    sparsity = 1 - (n_interactions / (n_users * n_items))
    print(f"Users: {n_users}, ")
    print(f"Items: {n_items}, ")
    print(f"Interactions: {n_interactions}")
    print(f"Sparsity: {sparsity}")
    print(f"Average sequence length: {round(float(seq_lengths.mean()), 2)}")
    print(f"Seq length percentiles: "
          f"p25={seq_lengths.quantile(0.25):.0f}, "
          f"p50={seq_lengths.quantile(0.50):.0f}, "
          f"p75={seq_lengths.quantile(0.75):.0f}, "
          f"p95={seq_lengths.quantile(0.95):.0f}")
    print(f"{dataset.head()}")


def compute_metadata(dataset, test, config):
    seq_lengths = dataset.groupby('user')['item'].count()
    n_users = dataset['user'].nunique()
    n_items = dataset['item'].nunique()
    n_interactions = len(dataset)
    sparsity = 1 - (n_interactions / (n_users * n_items))

    return {
        "dataset": config['dataset'],
        "threshold": config['threshold'],
        "kcore": config['kcore'],
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "sparsity": round(sparsity, 6),
        "avg_seq_length": round(float(seq_lengths.mean()), 2),
        "min_seq_length": int(seq_lengths.min()),
        "max_seq_length": int(seq_lengths.max()),
        "downstream_split_fraction": config['split'],
        "n_test_interactions": len(test) if test is not None else 0,
        "p25": seq_lengths.quantile(0.25),
        "p50": seq_lengths.quantile(0.50),
        "p75": seq_lengths.quantile(0.75),
        "p95": seq_lengths.quantile(0.95),
    }

def save(dataset, test, id2item, id2user, metadata, args):

    directory = os.path.join(RESULTS_DIR, args['dataset'])
    os.makedirs(directory, exist_ok=True)
    print(f"\nSaving to: {directory}.")

    with open(os.path.join(directory, 'id2user_mapping.pkl'), 'wb') as f:
        pickle.dump(id2user, f)

    with open(os.path.join(directory, 'id2item_mapping.pkl'), 'wb') as f:
        pickle.dump(id2item, f)

    with open(os.path.join(directory, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    if test is not None:
        test.to_csv(os.path.join(directory, f"{args['dataset']}_test.tsv"),
                       sep='\t', index=False, header=False)

    dataset.to_csv(os.path.join(directory, f"{args['dataset']}.tsv"),
                   sep='\t', index=False, header=False)


DATASET_CONFIGS = {
    'movielens': {'loader': load_movielens,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_beauty': {'loader': load_amazon_beauty,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_toys': {'loader': load_amazon_toys,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_books': {'loader': load_amazon_books,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_music': {'loader': load_amazon_music,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_cds': {'loader': load_amazon_cds,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_movies': {'loader': load_amazon_movies,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_sports': {'loader': load_amazon_sports,
                  'threshold': 0.0,
                  'kcore': 5,
                  'split': 0.2,
                  },
    'amazon_fashion': {'loader': load_amazon_fashion,
                      'threshold': 0.0,
                      'kcore': 5,
                      'split': 0.2,
                      },
    'amazon_clothing': {'loader': load_amazon_clothing,
                       'threshold': 0.0,
                       'kcore': 5,
                       'split': 0.2,
                       },
    'amazon_games': {'loader': load_amazon_games,
                        'threshold': 0.0,
                        'kcore': 5,
                        'split': 0.2,
                        },
}


if __name__ == '__main__':
    args = parser()

    if args['dataset'] not in DATASET_CONFIGS.keys():
        raise ValueError(f"Unknown dataset: {args['dataset']}. Available: {list(DATASET_CONFIGS.keys())}")

    dataset_name = args['dataset']
    config = DATASET_CONFIGS[dataset_name]
    kcore = args['kcore'] if args['kcore'] is not None else config['kcore']
    threshold = args['threshold'] if args['threshold'] is not None else config['threshold']
    split_fraction = args['split'] if args['split'] is not None else config['split']

    print(f"\nLoading dataset: {args['dataset']}")
    dataset = config['loader']()
    print_stats(dataset)

    print(f"\nBinarizing with threshold={threshold}...")
    dataset = binarize(dataset, threshold)
    print_stats(dataset)

    print(f"\nAppling {kcore}-core...")
    dataset = kcore_filter(dataset, kcore)
    print_stats(dataset)

    print(f"\nEncoding users and items...")
    dataset, id2user, id2item = encode_ids(dataset)
    print_stats(dataset)

    print(f"\nSplitting users...")
    dataset, test = split(dataset, split_fraction)
    print(f"Train set\n")
    print_stats(dataset)

    if test is not None:
        print("Test set\n")
        print_stats(test)

    print(f"\n--- Metadata ---\n")
    metadata = compute_metadata(dataset, test, {
    'dataset': dataset_name,
    'threshold': threshold,
    'kcore': kcore,
    'split': split_fraction,
    })

    print(f"Dataset: {metadata['dataset']}")
    print(f"Users: {metadata['n_users']}")
    print(f"Items: {metadata['n_items']}")
    print(f"Interactions: {metadata['n_interactions']}")
    print(f"Avg seq length: {metadata['avg_seq_length']}")
    print(f"Sparsity: {metadata['sparsity']}")
    print(f"Seq length percentiles: "
          f"p25={metadata['p25']:.0f}, "
          f"p50={metadata['p50']:.0f}, "
          f"p75={metadata['p75']:.0f}, "
          f"p95={metadata['p95']:.0f}")

    save(dataset, test, id2item, id2user, metadata, args)
    print("\nPreprocessing complete.")