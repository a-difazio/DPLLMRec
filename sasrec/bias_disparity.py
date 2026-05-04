import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--gen_name', required=True)
    return parser.parse_args()


def setup_paths(args):
    original_path = os.path.join('data', args.dataset + '.tsv')

    generated_path = os.path.join(
        'experiments',
        f'{args.dataset}_{args.run_name}',
        'generated_data',
        f"synthetic_{args.dataset}_{args.gen_name}.tsv"
    )

    if not os.path.exists(original_path):
        raise FileNotFoundError(original_path)

    if not os.path.exists(generated_path):
        raise FileNotFoundError(generated_path)

    return original_path, generated_path


def load_data(original_path, generated_path):
    original = pd.read_csv(original_path, sep="\t", names=['user', 'item'])
    generated = pd.read_csv(generated_path, sep="\t", names=['user', 'item'])

    return original, generated


def build_user_groups(df, n_groups=4):
    user_activity = df.groupby('user').size()

    # quartili
    bins = pd.qcut(user_activity, q=n_groups, labels=False, duplicates='drop')

    user_groups = bins.to_dict()  # user -> group_id

    return user_groups

def build_item_categories(df, n_groups=3):
    item_pop = df['item'].value_counts()

    bins = pd.qcut(item_pop, q=n_groups, labels=False, duplicates='drop')

    item_categories = bins.to_dict()  # item -> category

    return item_categories

def filter_known_items(df, item_categories):
    return df[df['item'].isin(item_categories)]

def assign_groups(df, user_groups, item_categories):
    df = df.copy()

    df['user_group'] = df['user'].map(user_groups)
    df['item_category'] = df['item'].map(item_categories)

    df = df.dropna(subset=['user_group', 'item_category'])

    return df

def compute_counts(df, n_user_groups, n_item_categories):
    category_sum = np.zeros((n_user_groups, n_item_categories))
    total_sum = np.zeros(n_user_groups)

    for _, row in df.iterrows():
        g = int(row['user_group'])
        c = int(row['item_category'])

        category_sum[g, c] += 1
        total_sum[g] += 1

    return category_sum, total_sum

def compute_PC(item_categories, n_item_categories):
    from collections import Counter

    counts = Counter(item_categories.values())

    total_items = len(item_categories)

    PC = np.array([
        counts.get(c, 0) / total_items
        for c in range(n_item_categories)
    ])

    return PC

def compute_bias(df, item_categories, n_user_groups, n_item_categories):
    category_sum, total_sum = compute_counts(df, n_user_groups, n_item_categories)
    total_sum[total_sum == 0] = 1e-12

    PC = compute_PC(item_categories, n_item_categories)
    PC[PC == 0] = 1e-12

    # P(C|G)
    P_C_given_G = (category_sum.T / total_sum).T  # broadcasting

    bias = P_C_given_G / PC

    return bias

def compute_bd(BS, BR):
    BS[BS == 0] = 1e-12
    return (BR - BS) / BS

def to_dataframe(matrix, user_label="G", item_label="C"):
    df = pd.DataFrame(matrix)

    df.index = [f"{user_label}{i}" for i in range(df.shape[0])]
    df.columns = [f"{item_label}{i}" for i in range(df.shape[1])]

    return df

def plot_heatmap(df, title, path):
    plt.figure(figsize=(6,5))
    plt.imshow(df, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    args = parse()
    original_path, generated_path = setup_paths(args)
    original, generated = load_data(original_path, generated_path)

    user_groups = build_user_groups(original)
    item_categories = build_item_categories(original)

    generated = filter_known_items(generated, item_categories)

    orig_df = assign_groups(original, user_groups, item_categories)
    gen_df = assign_groups(generated, user_groups, item_categories)

    n_user_groups = len(set(user_groups.values()))
    n_item_categories = len(set(item_categories.values()))

    BS = compute_bias(orig_df, item_categories, n_user_groups, n_item_categories)
    BR = compute_bias(gen_df, item_categories, n_user_groups, n_item_categories)

    BD = compute_bd(BS, BR)

    BS_df = to_dataframe(BS, "G", "C")
    BR_df = to_dataframe(BR, "G", "C")
    BD_df = to_dataframe(BD, "G", "C")

    output_dir = os.path.join(
        'experiments',
        f'{args.dataset}_{args.run_name}',
        'bias_disparity',
        args.gen_name
    )
    os.makedirs(output_dir, exist_ok=True)

    BS_df.to_csv(os.path.join(output_dir, "BS.csv"))
    BR_df.to_csv(os.path.join(output_dir, "BR.csv"))
    BD_df.to_csv(os.path.join(output_dir, "BD.csv"))

    plot_heatmap(BD_df, "Bias Disparity", os.path.join(output_dir, "BD.png"))