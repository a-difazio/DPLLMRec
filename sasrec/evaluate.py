import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, entropy
from scipy.spatial.distance import jensenshannon

from utils import *


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--gen_name', required=True)
    parser.add_argument('--seed', default=42)
    return parser.parse_args()


def setup_paths(args):
    output_dir = os.path.join('experiments', f'{args.dataset}_{args.run_name}', "generated_stats", args.gen_name)
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join('data', args.dataset + '.tsv')

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original dataset not found: {original_path}")

    generated_path = os.path.join('experiments', f'{args.dataset}_{args.run_name}', 'generated_data',
                                  f"synthetic_{args.dataset}_{args.gen_name}.tsv")

    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated dataset not found: {generated_path}")

    return output_dir, original_path, generated_path

def popularity_entropy(pop):
    probs = pop.values
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

# Gini Coefficient
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)

    array = array + 1e-12
    array = np.sort(array)

    n = len(array)
    index = np.arange(1, n + 1)

    return np.sum((2 * index - n - 1) * array) / (n * np.sum(array))

    # Head Overlap
    # quanto coincidono gli item popolari
def head_overlap(pop_orig, pop_gen, k=50):

    top_orig = set(pop_orig.sort_values(ascending=False).head(k).index)
    top_gen = set(pop_gen.sort_values(ascending=False).head(k).index)

    overlap = len(top_orig.intersection(top_gen)) / k
    return overlap

def get_top_transitions(df, top_k=20):
    transitions = df.dropna(subset=['prev_item']).copy()

    transitions['pair'] = transitions['prev_item'].astype(str) + " -> " + transitions['item'].astype(int).astype(str)

    counts = transitions['pair'].value_counts(normalize=True)
    return counts, set(counts.head(top_k).index)


def count_3_cycles(df):
    df = df.copy()

    df['prev3_item'] = df.groupby('user')['item'].shift(3)

    cycles = df[df['item'] == df['prev3_item']]

    return len(cycles) / len(df)


def count_short_loops(df):
    df = df.copy()

    df['prev_item'] = df.groupby('user')['item'].shift(1)
    df['prev2_item'] = df.groupby('user')['item'].shift(2)

    loops = df[(df['item'] == df['prev2_item'])]

    return len(loops) / len(df)

def ngram_distribution(df, n=2, top_k=None):
    ngrams = []
    for _, group in df.groupby('user'):
        items = group['item'].tolist()
        for i in range(len(items) - n + 1):
            ngrams.append(tuple(items[i:i+n]))
    counts = pd.Series(ngrams).value_counts(normalize=True)
    return counts.head(top_k) if top_k else counts


def ngram_jsd(df_orig, df_gen, n=2, top_k=None):
    dist_orig = ngram_distribution(df_orig, n, top_k)
    dist_gen  = ngram_distribution(df_gen, n, top_k)
    all_ngrams = set(dist_orig.index) | set(dist_gen.index)
    dist_orig  = dist_orig.reindex(all_ngrams, fill_value=0)
    dist_gen   = dist_gen.reindex(all_ngrams, fill_value=0)
    epsilon = 1e-12
    return float(jensenshannon(dist_orig + epsilon, dist_gen + epsilon))


def compute_stats(original_path, generated_path):
    original = pd.read_csv(original_path, sep="\t", names=['user', 'item'])
    generated = pd.read_csv(generated_path, sep="\t", names=['user', 'item'])

    stats = {}

    # Basic stats
    print("\n--- Basic statistics ---\n")

    stats['original_users'] = original['user'].nunique()
    stats['original_items'] = original['item'].nunique()
    stats['original_interactions'] = len(original)

    stats['generated_users'] = generated['user'].nunique()
    stats['generated_items'] = generated['item'].nunique()
    stats['generated_interactions'] = len(generated)


    print("Original")
    print(f"Number of users: {stats['original_users']}")
    print(f"Number of items: {stats['original_items']}")
    print(f"Number of interactions: {stats['original_interactions']}")

    print("\nGenerated")
    print(f"Number of users: {stats['generated_users']}")
    print(f"Number of items: {stats['generated_items']}")
    print(f"Number of interactions: {stats['generated_interactions']}")

    # Coverage
    print("\n--- Item Coverage ---\n")

    original_items = set(original['item'].unique())
    generated_items = set(generated['item'].unique())
    stats['coverage'] = len(generated_items.intersection(original_items)) / len(original_items)

    print(f"Coverage: {stats['coverage']:.4f}")

    # Popularity Bias
    all_items = original_items | generated_items
    pop_original = original['item'].value_counts(normalize=True).reindex(all_items, fill_value=0)
    pop_generated = generated['item'].value_counts(normalize=True).reindex(all_items, fill_value=0)

    # Top-k items
    print("\n--- Top 50 most frequent items ---\n")
    print(f"Original: {pop_original.sort_values(ascending=False).head(50).index.tolist()}")
    print(f"Generated: {pop_generated.sort_values(ascending=False).head(50).index.tolist()}")

    # Max Frequency
    print("\n--- Max Frequency ---\n")
    stats['max_item_freq_orig'] = float(pop_original.max())
    stats['max_item_freq'] = float(pop_generated.max())

    print(f"Original: {stats['max_item_freq_orig']}")
    print(f"Generated: {stats['max_item_freq']}")


    # Distribution Metrics
    print("\n--- Distribution ---\n")
    stats['_pop_original'] = pop_original.sort_index()
    stats['_pop_generated'] = pop_generated.sort_index()

    epsilon = 1e-12
    stats['kl_divergence'] = entropy(stats['_pop_original'] + epsilon, stats['_pop_generated'] + epsilon)
    stats['js_divergence'] = jensenshannon(stats['_pop_original'] + epsilon, stats['_pop_generated'] + epsilon)
    stats['entropy_orig'] = popularity_entropy(stats['_pop_original'])
    stats['entropy_gen'] = popularity_entropy(stats['_pop_generated'])
    stats['gini_orig'] = gini(stats['_pop_original'].values)
    stats['gini_gen'] = gini(stats['_pop_generated'].values)
    stats['head_overlap_20'] = head_overlap(stats['_pop_original'], stats['_pop_generated'], 20)
    stats['head_overlap_50'] = head_overlap(stats['_pop_original'], stats['_pop_generated'], 50)

    print(f"KL Divergence (Original || Generated): {stats['kl_divergence']:.4f}")

    # se original > gen (popolarity collapse)
    # se gen > original (distribuzione più uniforme)
    print(f"Popularity entropy (original): {stats['entropy_orig']:.4f}")
    print(f"Popularity entropy (generated): {stats['entropy_gen'] :.4f}")

    # gen > original pochi item dominano
    # original > gen distribuzione più piatta
    print(f"Gini coefficient (original): {stats['gini_orig']:.4f}")
    print(f"Gini coefficient (generated): {stats['gini_gen']:.4f}")

    print(f"Head overlap@20: {stats['head_overlap_20']:.4f}")
    print(f"Head overlap@50: {stats['head_overlap_50']:.4f}")

    # Rank correlation (Spearman e Kendall)
    rank_orig = pop_original.rank(ascending=False)
    rank_gen = pop_generated.rank(ascending=False)
    stats['spearman_corr'], _ = spearmanr(rank_orig, rank_gen)
    stats['kendall_corr'], _ = kendalltau(rank_orig, rank_gen)

    print(f"Spearman rank correlation: {stats['spearman_corr']:.4f}")
    print(f"Kendall tau correlation: {stats['kendall_corr']:.4f}")

    # Sequence length
    print(f"\n--- Sequence length ---\n")

    stats['_len_orig'] = original.groupby('user').size()
    stats['avg_len_orig'] = float(stats['_len_orig'].mean())
    stats['min_len_orig'] = int(stats['_len_orig'].min())
    stats['max_len_orig'] = int(stats['_len_orig'].max())

    print('Original')
    print(f"Avg length: {stats['avg_len_orig']:.4f}")
    print(f"Min length: {stats['min_len_orig']:.4f}")
    print(f"Max length: {stats['max_len_orig']:.4f}")

    stats['_len_gen']  = generated.groupby('user').size()
    stats['avg_len_gen'] = float(stats['_len_gen'].mean())
    stats['min_len_gen'] = int(stats['_len_gen'].min())
    stats['max_len_gen'] = int(stats['_len_gen'].max())

    print('Generated')
    print(f"Avg length: {stats['avg_len_gen']:.4f}")
    print(f"Min length: {stats['min_len_gen']:.4f}")
    print(f"Max length: {stats['max_len_gen']:.4f}")

    common = stats['_len_orig'].index.intersection(stats['_len_gen'].index)
    stats['same_len_perc'] = (stats['_len_orig'].loc[common] == stats['_len_gen'].loc[common]).mean() * 100

    print(f"Same length: {stats['same_len_perc']}")

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

    print("\n--- Repeat item distribution (top 10) ---")
    repeat_items = repeats['item'].value_counts(normalize=True)
    print(repeat_items.head(10))

    stats['_repeat_items'] = repeat_items
    stats['rep_rate_orig'] = float(rep_rate_orig)
    stats['rep_rate_gen'] = float(rep_rate_gen)
    stats['top5_repeat_share'] = float(repeat_items.head(5).sum())

    # Repetition of Popular Items
    top_pop_items = set(pop_generated.sort_values(ascending=False).head(50).index)
    repeat_top_items = set(repeat_items.head(50).index)

    stats['repeat_popularity_overlap'] = len(top_pop_items & repeat_top_items) / len(repeat_top_items)
    print(f"Overlap between repeated items and top popularity: {stats['repeat_popularity_overlap']:.2f}")

    top5_share = repeat_items.head(5).sum()
    print(f"Top-5 items explain {top5_share:.2%} of repetitions")

    stats['short_loops_orig'] = float(count_short_loops(original))
    stats['short_loops_gen'] = float(count_short_loops(generated))

    print(f"Short loops (original): {stats['short_loops_orig']:.4f}")
    print(f"Short loops (generated): {stats['short_loops_gen']:.4f}")

    # 3-cycles
    stats['cycles3_orig'] = float(count_3_cycles(original))
    stats['cycles3_gen'] = float(count_3_cycles(generated))

    print(f"3-cycles (original): {stats['cycles3_orig']:.4f}")
    print(f"3-cycles (generated): {stats['cycles3_gen']:.4f}")

    # Transition
    print(f"\n--- Transition Analysis ---\n")

    trans_orig_dist, top_k_orig = get_top_transitions(original_copy, top_k=50)
    trans_gen_dist, top_k_gen = get_top_transitions(generated_copy, top_k=50)

    print("\nTop 5 real transition:")
    print(trans_orig_dist.head(5))
    print("\nTop 5 generated transition:")
    print(trans_gen_dist.head(5))

    # Transition overlap
    intersection = len(top_k_orig.intersection(top_k_gen))
    union = len(top_k_orig.union(top_k_gen))

    print(f"Overlap on Top-50 bigrams: {intersection}/50 transition are the same.")

    # Jaccard Similarity
    jaccard = intersection / union if union > 0 else 0
    print(f"Jaccard Similarity (Top-50): {jaccard:.4f}")

    # ngram, loops
    stats['jsd_bigram'] = ngram_jsd(original, generated, n=2)
    stats['jsd_trigram'] = ngram_jsd(original, generated, n=3)

    print(f"Bigram JSD:  {stats['jsd_bigram']:.4f}")
    print(f"Trigram JSD: {stats['jsd_trigram']:.4f}")

    return stats

def save_stats(stats, output_dir, gen_name):

    scalar_stats = {k: v for k, v in stats.items() if not k.startswith('_')}

    json_path = os.path.join(output_dir, f"stats_{gen_name}.json")
    with open(json_path, 'w') as f:
        json.dump(scalar_stats, f, indent=2)

    print(f"Stats saved to {json_path}", flush=True)


def save_plots(stats, output_dir, gen_name):
    pop_orig = stats['_pop_original']
    pop_gen = stats['_pop_generated']
    len_orig = stats['_len_orig']
    len_gen = stats['_len_gen']
    repeat_items = stats['_repeat_items']

    # Popularity distribution
    sorted_orig = pop_orig.sort_values(ascending=False).values
    sorted_gen = pop_gen.sort_values(ascending=False).values

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_orig, label='Original Data', color='blue', alpha=0.7)
    plt.plot(sorted_gen, label='Generated Data', color='red', alpha=0.5)
    plt.yscale('log')
    plt.title('Item Popularity Distribution (Log Scale)')
    plt.xlabel('Item Rank')
    plt.ylabel('Frequency (Normalized)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, f"popularity_distribution_{gen_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


    # Box Plot Frequency Top-k Items
    original_top_items = pop_orig.sort_values(ascending=False).index[:50]

    freq_orig = pop_orig[original_top_items].values
    freq_gen = pop_gen[original_top_items].values
    bar_width = 0.4
    x = np.arange(len(original_top_items))
    plt.figure(figsize=(12,6))
    plt.bar(x - bar_width/2, freq_orig, width=bar_width, label='Original', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, freq_gen, width=bar_width, label='Generated', color='red', alpha=0.5)
    plt.xlabel('Item')
    plt.ylabel('Frequency')
    plt.title('Top 50 original items frequency: Original vs Generated')
    plt.legend()
    plt.xticks(x, original_top_items, rotation=90)
    plt.savefig(os.path.join(output_dir, f"popularity_original_{gen_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    generated_top_items = pop_gen.sort_values(ascending=False).index[:50]

    freq_orig = pop_orig[generated_top_items].values
    freq_gen = pop_gen[generated_top_items].values
    bar_width = 0.4
    x = np.arange(len(generated_top_items))
    plt.figure(figsize=(12,5))
    plt.bar(x - bar_width/2, freq_orig, width=bar_width, label='Original', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, freq_gen, width=bar_width, label='Generated', color='red', alpha=0.5)
    plt.xlabel('Item')
    plt.ylabel('Frequency')
    plt.title('Top 50 generated items frequency: Original vs Generated')
    plt.legend()
    plt.xticks(x, generated_top_items, rotation=90)
    plt.savefig(os.path.join(output_dir, f"popularity_generated_{gen_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()


    # Lorenz Curve
    def lorenz_curve(values):

        sorted_vals = np.sort(values)
        cumvals = np.cumsum(sorted_vals)

        cumvals = np.insert(cumvals, 0, 0)
        cumvals = cumvals / cumvals[-1]

        x = np.linspace(0,1,len(cumvals))

        return x, cumvals

    x_o, y_o = lorenz_curve(pop_orig.values)
    x_g, y_g = lorenz_curve(pop_gen.values)

    plt.plot(x_o, y_o, label="Original")
    plt.plot(x_g, y_g, label="Generated")
    plt.plot([0,1],[0,1], linestyle="--", color="black")
    plt.title("Lorenz Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir,f"lorenz_curve_{gen_name}.png"))
    plt.close()

    # Frequency Scatter
    plt.figure(figsize=(12,12))
    plt.scatter(pop_orig.values, pop_gen.values, alpha=0.3)
    max_val = max(pop_orig.max(), pop_gen.max())
    plt.plot([0,max_val],[0,max_val], color='red')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Original frequency")
    plt.ylabel("Generated frequency")
    plt.title("Item frequency scatter")
    plt.savefig(os.path.join(output_dir, f"frequency_scatter_{gen_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # CDF
    cdf_orig = np.cumsum(np.sort(pop_orig.values)) / np.sum(pop_orig.values)
    cdf_gen = np.cumsum(np.sort(pop_gen.values)) / np.sum(pop_gen.values)
    plt.figure(figsize=(12,6))
    plt.plot(cdf_orig, label='Original', color='blue', alpha=0.7)
    plt.plot(cdf_gen, label='Generated', color='red', alpha=0.5)
    plt.xlabel('Item rank (sorted by frequency)')
    plt.ylabel('CDF')
    plt.title('CDF of item frequencies')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"cdf_{gen_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Length Distribution
    max_val = max(len_orig.max(), len_gen.max())
    bins = np.linspace(0, max_val, 30)
    plt.figure(figsize=(12,6))
    plt.hist([len_orig, len_gen], bins=bins,  color=['blue', 'red'],
             label=['Original', 'Generated'], density=True, alpha=0.7, edgecolor='black')
    plt.grid(axis='y', alpha=0.3)
    plt.title('Session Length Distribution')
    plt.xlabel('Item per user')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"length_distribution_{gen_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Repeat items
    plt.figure(figsize=(12, 6))
    repeat_items.head(20).plot(kind="bar")
    plt.title("Top 20 repeated items")
    plt.ylabel("Fraction of repetitions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"repeat_items_{gen_name}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Repeat vs popularity
    pop_rank = pop_gen.rank(ascending=False)
    repeat_prob = repeat_items.reindex(pop_gen.index).fillna(0)
    plt.figure(figsize=(12, 6))
    plt.scatter(pop_rank, repeat_prob, alpha=0.3)
    plt.xlabel('Popularity rank')
    plt.ylabel('Repeat probability')
    plt.title('Repeat probability vs Popularity rank')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"repeat_vs_popularity_{gen_name}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    args = parse()
    set_seed(args.seed)
    output_dir, original_path, generated_path = setup_paths(args)

    print('Original:', original_path)
    print('Generated:', generated_path)
    print('Output directory:', output_dir)

    stats = compute_stats(original_path, generated_path)
    save_stats(stats, output_dir, args.gen_name)
    save_plots(stats, output_dir, args.gen_name)










