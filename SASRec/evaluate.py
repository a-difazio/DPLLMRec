import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SASRec.utils import data_partition
from scipy.stats import spearmanr, kendalltau, entropy



parser = argparse.ArgumentParser()
parser.add_argument('--original_dataset', required=True)
parser.add_argument('--generated_dir', required=True)
args = parser.parse_args()

output_dir = os.path.join(args.generated_dir, "results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_path = os.path.join(output_dir, "evaluation_log.txt")
log_file = open(log_path, "w")


def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


original_path = os.path.join('data', args.original_dataset + '.txt')
generated_path = os.path.join(args.generated_dir, 'generated.txt')

log('Original:', original_path)
log('Generated:', generated_path)

original = pd.read_csv(original_path, sep=" ", names=['user', 'item'])
generated = pd.read_csv(generated_path, sep=",", names=['user', 'item'])

# TODO: generate on all the dataset
dataset = data_partition('ml-1m')
[user_train, user_valid, user_test, usernum, itemnum] = dataset

df_train = (
    pd.DataFrame.from_dict(user_train, orient='index')
      .stack()
      .reset_index()
)

df_train.columns = ['user', 'seq_pos', 'item']
df_train.drop('seq_pos', axis=1, inplace=True)
df_train['item'] = df_train['item'].astype(int)
original = df_train


# len, nusers, nitems
log("\n--- Basic statistics ---\n")
log("Original")
log(f"Dataset length: {len(original)}")
log(f"Number of users: {original['user'].nunique()}")
log(f"Number of items: {original['item'].nunique()}")

log("\nGenerated")
log(f"Dataset length: {len(generated)}")
log(f"Number of users: {generated['user'].nunique()}")
log(f"Number of items: {generated['item'].nunique()}")


# item coverage
log("\n--- Item Coverage ---\n")
original_items = set(original['item'].unique())
generated_items = set(generated['item'].unique())
coverage = len(generated_items.intersection(original_items)) / len(original_items)

log(f"Coverage: {coverage:.4f}")

# popularity bias
pop_original = original['item'].value_counts(normalize=True)
pop_generated = generated['item'].value_counts(normalize=True)

# missing items
all_items = list(original_items)
missing_items = [item for item in all_items if item not in pop_generated.index]
pop_generated = pd.concat([pop_generated, pd.Series(0, index=missing_items)])

all_items = list(generated_items)
missing_items = [item for item in all_items if item not in pop_original.index]
pop_original = pd.concat([pop_original, pd.Series(0, index=missing_items)])

only_in_orig = set(pop_original.index) - set(pop_generated.index)
# log("Item only in original:", only_in_orig, len(only_in_orig))
only_in_gen = set(pop_generated.index) - set(pop_original.index)
# log("Item only in generated:", only_in_gen, len(only_in_gen))


sorted_items_org = pop_original.sort_values(ascending=False).index
sorted_items_gen = pop_generated.sort_values(ascending=False).index

plt.figure(figsize=(12, 6))
plt.plot(pop_original[sorted_items_org].values, label='Original Data', color='blue', alpha=0.7)
plt.plot(pop_generated[sorted_items_gen].values, label='Generated Data', color='red', alpha=0.5)
plt.yscale('log')
plt.title('Item Popularity Distribution (Log Scale)')
plt.xlabel('Item Rank')
plt.ylabel('Frequency (Normalized)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(os.path.join(output_dir, "popularity_distribution_plot.png"), dpi=200, bbox_inches="tight")
plt.close()


# top 50 items
log("\n--- Top 20 most frequent items ---\n")
log(f"Original: {pop_original.head(50).index.tolist()}")
log(f"Generated: {pop_generated.head(50).index.tolist()}")


# Box plot frequency top items
original_top_items = pop_original.sort_values(ascending=False).index[:50]

freq_orig = pop_original[original_top_items].values
freq_gen = pop_generated[original_top_items].values

bar_width = 0.4
x = np.arange(len(original_top_items))

plt.figure(figsize=(15,5))
plt.bar(x - bar_width/2, freq_orig, width=bar_width, label='Original', color='blue', alpha=0.7)
plt.bar(x + bar_width/2, freq_gen, width=bar_width, label='Generated', color='red', alpha=0.5)
plt.xlabel('Item')
plt.ylabel('Frequency')
plt.title('Top 20 original items frequency: Original vs Generated')
plt.legend()
plt.xticks(x, original_top_items, rotation=90)
plt.savefig(os.path.join(output_dir, "popularity_original_barplot.png"), dpi=200, bbox_inches="tight")
plt.close()

generated_top_items = pop_generated.sort_values(ascending=False).index[:50]

freq_orig = pop_original[generated_top_items].values
freq_gen = pop_generated[generated_top_items].values

bar_width = 0.4
x = np.arange(len(generated_top_items))

plt.figure(figsize=(15,5))
plt.bar(x - bar_width/2, freq_orig, width=bar_width, label='Original', color='blue', alpha=0.7)
plt.bar(x + bar_width/2, freq_gen, width=bar_width, label='Generated', color='red', alpha=0.5)
plt.xlabel('Item')
plt.ylabel('Frequency')
plt.title('Top 20 generated items frequency: Original vs Generated')
plt.legend()
plt.xticks(x, generated_top_items, rotation=90)
plt.savefig(os.path.join(output_dir, "popularity_generated_barplot.png"), dpi=200, bbox_inches="tight")
plt.close()

# Distribution
log("\n--- Distribution ---\n")
pop_generated = pop_generated.sort_index()
pop_original = pop_original.sort_index()

epsilon = 1e-12
kl_div = entropy(pop_original + epsilon, pop_generated + epsilon)
log(f"KL Divergence (Original || Generated): {kl_div:.4f}")

# CDF
cdf_orig = np.cumsum(np.sort(pop_original.values))
cdf_gen = np.cumsum(np.sort(pop_generated.values))

plt.figure(figsize=(8,5))
plt.plot(cdf_orig, label='Original', color='blue', alpha=0.7)
plt.plot(cdf_gen, label='Generated', color='red', alpha=0.5)
plt.xlabel('Item rank (sorted by frequency)')
plt.ylabel('CDF')
plt.title('CDF of item frequencies')
plt.legend()
plt.savefig(os.path.join(output_dir, "cdf_plot.png"), dpi=200, bbox_inches="tight")
plt.close()

# Rank correlation (Spearman e Kendall)
rank_orig = pop_original.rank(ascending=False)
rank_gen = pop_generated.rank(ascending=False)

spearman_corr, _ = spearmanr(rank_orig, rank_gen)
kendall_corr, _ = kendalltau(rank_orig, rank_gen)

log(f"Spearman rank correlation: {spearman_corr:.4f}")
log(f"Kendall tau correlation: {kendall_corr:.4f}")

log("\n--- Length Distribution ---\n")

# Sequence Length Distribution

len_orig = original.groupby('user').size()
len_gen = generated.groupby('user').size()

max_val = max(len_orig.max(), len_gen.max())
bins = np.linspace(0, max_val, 30)
plt.figure(figsize=(15,5))
plt.hist([len_orig, len_gen],
         bins=bins,
         color=['blue', 'red'],
         label=['Original', 'Generated'],
         density=True,
         alpha=0.7,
         edgecolor='black')
plt.grid(axis='y', alpha=0.3)
plt.title('Session Length Distribution')
plt.xlabel('Item for user')
plt.legend()
plt.savefig(os.path.join(output_dir, "length_distribution_plot.png"), dpi=200, bbox_inches="tight")
plt.close()


# Same session length original vs generated

common = len_orig.index.intersection(len_gen.index)
same_length = (len_orig.loc[common] == len_gen.loc[common])
percent = same_length.mean() * 100
count_same = same_length.sum()
total = len(common)

log(f"{count_same}/{total} users ({percent:.2f}%) have the same session length as in the original dataset.")


# Repetition
log("\n--- Repetition ---\n")

# Shift of 1 item
original_copy = original.copy()
original_copy['prev_item'] = original_copy.groupby('user')['item'].shift(1)
original_copy['prev_item'] = original_copy['prev_item']

repeats = original_copy[original_copy['item'] == original_copy['prev_item']]
rep_rate_orig = len(repeats) / len(original_copy)
n_rep_orig = len(repeats)
log(f"Original: {rep_rate_orig:.4%} ({n_rep_orig} repetition)")

generated_copy = generated.copy()
generated_copy['prev_item'] = generated_copy.groupby('user')['item'].shift(1)
generated_copy['prev_item'] = generated_copy['prev_item']

repeats = generated_copy[generated_copy['item'] == generated_copy['prev_item']]
rep_rate_gen = len(repeats) / len(generated_copy)
n_rep_gen = len(repeats)
log(f"Generated: {rep_rate_gen:.4%} ({n_rep_gen} repetition)")

# TODO: Short Loops

# Transition
print(f"\n--- Transition Analysis ---\n")


def get_top_transitions(df, top_k=20):
    transitions = df.dropna(subset=['prev_item']).copy()

    transitions['pair'] = transitions['prev_item'].astype(str) + " -> " + transitions['item'].astype(int).astype(str)

    counts = transitions['pair'].value_counts(normalize=True)
    return counts, set(counts.head(top_k).index)


trans_orig_dist, top_k_orig = get_top_transitions(original_copy, top_k=100)
trans_gen_dist, top_k_gen = get_top_transitions(generated_copy, top_k=100)

# Jaccard Similarity
intersection = len(top_k_orig.intersection(top_k_gen))
union = len(top_k_orig.union(top_k_gen))
jaccard = intersection / union if union > 0 else 0


print(f"Overlap on Top-50 bigrams: {intersection}/50 transition are the same.")
print(f"Jaccard Similarity (Top-50): {jaccard:.4f}")

print("\nTop 5 real transition:")
print(trans_orig_dist.head(5))
print("\nTop 5 generated transition:")
print(trans_gen_dist.head(5))

log_file.close()





