import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, entropy


parser = argparse.ArgumentParser()
parser.add_argument('--original_dataset', required=True)
parser.add_argument('--generated_dir', required=True)
args = parser.parse_args()

output_dir = os.path.join(args.generated_dir, "results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

original_path = os.path.join('data', args.original_dataset + '.txt')
generated_path = os.path.join(args.generated_dir, 'generated.txt')

original = pd.read_csv(original_path, sep=" ", names=['user', 'item'])
generated = pd.read_csv(generated_path, sep=",", names=['user', 'item'])

# len, nusers, nitems
print("\n--- Basic statistics ---\n")
print("Original")
print(f"Dataset length: {len(original)}")
print(f"Number of users: {original['user'].nunique()}")
print(f"Number of items: {original['item'].nunique()}")

print("\nGenerated")
print(f"Dataset length: {len(generated)}")
print(f"Number of users: {generated['user'].nunique()}")
print(f"Number of items: {generated['item'].nunique()}")


# item coverage
print("\n--- Item Coverage ---\n")
original_items = set(original['item'].unique())
generated_items = set(generated['item'].unique())
coverage = len(generated_items.intersection(original_items)) / len(original_items)

print(f"Coverage: {coverage:.4f}")

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
# print("Item only in original:", only_in_orig, len(only_in_orig))
only_in_gen = set(pop_generated.index) - set(pop_original.index)
# print("Item only in generated:", only_in_gen, len(only_in_gen))


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
print("\n--- Top 20 most frequent items ---\n")
print(f"Original: {pop_original.head(50).index.tolist()}")
print(f"Generated: {pop_generated.head(50).index.tolist()}")


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
print("\n--- Distribution ---\n")
pop_generated = pop_generated.sort_index()
pop_original = pop_original.sort_index()

epsilon = 1e-12
kl_div = entropy(pop_original + epsilon, pop_generated + epsilon)
print(f"KL Divergence (Original || Generated): {kl_div:.4f}")

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

print(f"Spearman rank correlation: {spearman_corr:.4f}")
print(f"Kendall tau correlation: {kendall_corr:.4f}")


# Sequence Length Distribution


# Repetition
print("\n--- Repetition ---\n")

# Shift of 1 item
original_copy = original.copy()
original_copy['prev_item'] = original_copy.groupby('user')['item'].shift(1)
original_copy['prev_item'] = original_copy['prev_item']

repeats = original_copy[original_copy['item'] == original_copy['prev_item']]
rep_rate_orig = len(repeats) / len(original_copy)
n_rep_orig = len(repeats)
print(f"Original: {rep_rate_orig:.4%} ({n_rep_orig} repetition)")

generated_copy = generated.copy()
generated_copy['prev_item'] = generated_copy.groupby('user')['item'].shift(1)
generated_copy['prev_item'] = generated_copy['prev_item']

repeats = generated_copy[generated_copy['item'] == generated_copy['prev_item']]
rep_rate_gen = len(repeats) / len(generated_copy)
n_rep_gen = len(repeats)
print(f"Generated: {rep_rate_gen:.4%} ({n_rep_gen} repetition)")

# Short Loops





