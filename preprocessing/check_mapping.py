from preprocess import kcore_filter, binarize
import pandas as pd
import pickle
import os

def processed_dataset(path, dataset):
    processed_train = pd.read_csv(os.path.join(path, f'{dataset}.tsv'), sep='\t', names=['user', 'item'])
    processed_test = pd.read_csv(os.path.join(path, f'{dataset}_test.tsv'), sep='\t', names=['user', 'item'])
    processed = pd.concat([processed_train, processed_test], ignore_index=True)

    with open(os.path.join(path,'id2item_mapping.pkl'), 'rb') as f:
        id2item = pickle.load(f)

    with open(os.path.join(path, 'id2user_mapping.pkl'), 'rb') as f:
        id2user = pickle.load(f)

    processed['original_item'] = processed['item'].map(id2item)
    processed['original_user'] = processed['user'].map(id2user)

    return processed

def raw_dataset(dataset):
    raw = pd.read_csv(
        os.path.join('raw', f'{dataset}.csv'),
        names=['item', 'user', 'rating', 'timestamp']
    )

    downloaded = pd.read_csv(
        os.path.join('2018', f'{dataset}.csv'),
        names=['item', 'user', 'rating', 'timestamp']
    )

    print("Same raw and downalod:", raw.equals(downloaded))

    raw_filtered = binarize(raw, 0.0)
    raw_filtered = kcore_filter(raw_filtered, 5)

    metadata = pd.read_json( os.path.join('2018', f'meta_{dataset}.json'), lines=True)
    metadata = metadata[['asin', 'title']]

    return raw_filtered, metadata

def raw_movielens():
    raw = pd.read_csv(
        os.path.join('raw', 'ratings.dat'),
        sep='::',
        names=['user', 'item', 'rating', 'timestamp'],
        engine='python'
    )

    downloaded = pd.read_csv(
        os.path.join('2018', 'ratings.dat'),
        sep='::',
        names=['user', 'item', 'rating', 'timestamp'],
        engine='python'
    )

    print("Same raw and downalod:", raw.equals(downloaded))

    raw_filtered = binarize(raw, 0.0)
    raw_filtered = kcore_filter(raw_filtered, 5)


    metadata = pd.read_csv(
        os.path.join('2018', 'movies.dat'),
        sep='::',
        names=['asin', 'title', 'genres'],
        engine='python',
        encoding="latin1"
    )
    metadata = metadata[['asin', 'title']]

    return raw_filtered, metadata

def check_dataset(processed, raw_filtered, metadata):
    raw_items = set(raw_filtered['item'].unique())
    decoded_items = set(processed['original_item'].unique())

    print("Raw Filtered items:", raw_filtered['item'].nunique())
    print("Encoded items:", processed['item'].nunique())
    print("Decoded items:", processed['original_item'].nunique())
    print(len(set(raw_filtered['item']) & set(processed['original_item'])))

    print("Missing in decoded:", len(raw_items - decoded_items))
    print("Extra in decoded:", len(decoded_items - raw_items))
    print("Len processed:", len(processed))
    print("Len raw filtered:", len(raw_filtered))

    print(list(processed['original_item'].unique())[10])
    print(list(metadata['asin'].unique())[10])

    print(processed['original_item'].head())
    print(metadata['asin'].head())

    merged = processed.merge(
        metadata,
        left_on='original_item',
        right_on='asin',
        how='left'
    )

    print(merged[['item', 'original_item', 'title']].head(20))

    missing_titles = merged['title'].isna().sum()
    present_titles = merged['title'].notna().sum()

    print("Missing titles:", missing_titles)
    print("Present titles:", present_titles)

    items_with_metadata = merged[merged['title'].notna()]['original_item'].nunique()

    items_without_metadata = merged[merged['title'].isna()]['original_item'].nunique()

    print("Items with metadata:", items_with_metadata)
    print("Items without metadata:", items_without_metadata)

    remaining_interactions = merged['title'].notna().sum()
    lost_interactions = merged['title'].isna().sum()

    print("Remaining interactions:", remaining_interactions)
    print("Lost interactions:", lost_interactions)

    coverage_items = (
            merged[merged['title'].notna()]['original_item'].nunique()
            / merged['original_item'].nunique()
    )

    print("Item coverage:", round(coverage_items * 100, 2), "%")

    coverage_interactions = (
            merged['title'].notna().sum()
            / len(merged)
    )

    print("Interaction coverage:", round(coverage_interactions * 100, 2), "%")

print('--- Amazon Games ---')
processed = processed_dataset('preprocessed/amazon_games', 'amazon_games')
raw_filtered, metadata = raw_dataset('Video_Games')
check_dataset(processed, raw_filtered, metadata)

print('--- Amazon CDs ---')
processed = processed_dataset('preprocessed/amazon_cds', 'amazon_cds')
raw_filtered, metadata = raw_dataset('CDs_and_Vinyl')
check_dataset(processed, raw_filtered, metadata)

print('--- Amazon Music ---')
processed = processed_dataset('preprocessed/amazon_music', 'amazon_music')
raw_filtered, metadata = raw_dataset('Digital_Music')
check_dataset(processed, raw_filtered, metadata)

print('--- Movielens ---')
processed = processed_dataset('preprocessed/movielens', 'movielens')
raw_filtered, metadata = raw_movielens()
check_dataset(processed, raw_filtered, metadata)