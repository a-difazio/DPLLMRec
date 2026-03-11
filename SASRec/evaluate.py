import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, entropy

from utils import *


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--generated', required=True)
    parser.add_argument('--seed', default=42)
    return parser.parse_args()


def setup_paths(args):
    output_dir = os.path.join('experiments', args.experiment, "dataset_comparison", args.generated)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    original_path = os.path.join('data', args.dataset + '.txt')

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original dataset not found: {original_path}")

    generated_path = os.path.join('experiments', f'{args.experiment}', 'generated_data',
                                  f"{args.generated}.txt")

    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated dataset not found: {generated_path}")

    return output_dir, original_path, generated_path


def setup_logs(output_dir):
    log_path = os.path.join(output_dir, "log.txt")
    logger = open(log_path, "w")

    return logger


def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=logger)


if __name__ == '__main__':

    args = parse()
    set_seed(args.seed)
    output_dir, original_path, generated_path = setup_paths(args)
    logger = setup_logs(output_dir)


    log('Original:', original_path)
    log('Generated:', generated_path)
    log('Output directory:', output_dir)

    original = pd.read_csv(original_path, sep=" ", names=['user', 'item'])
    generated = pd.read_csv(generated_path, sep=" ", names=['user', 'item'])


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

    # Item Coverage
    log("\n--- Item Coverage ---\n")
    original_items = set(original['item'].unique())
    generated_items = set(generated['item'].unique())
    coverage = len(generated_items.intersection(original_items)) / len(original_items)
    log(f"Coverage: {coverage:.4f}")

    # Popularity Bias Plot
    pop_original = original['item'].value_counts(normalize=True)
    pop_generated = generated['item'].value_counts(normalize=True)

    # Missing Items
    all_items = list(original_items)
    missing_items = [item for item in all_items if item not in pop_generated.index]
    print(missing_items)
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


    # Top-k items
    log("\n--- Top 50 most frequent items ---\n")
    log(f"Original: {pop_original.head(50).index.tolist()}")
    log(f"Generated: {pop_generated.head(50).index.tolist()}")


    # Box Plot Grequency Top-k Items
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

    # Distribution Metrics
    log("\n--- Distribution ---\n")
    pop_generated = pop_generated.sort_index()
    pop_original = pop_original.sort_index()

    epsilon = 1e-12
    kl_div = entropy(pop_original + epsilon, pop_generated + epsilon)
    log(f"KL Divergence (Original || Generated): {kl_div:.4f}")

    # Popularity Entropy
    # quanto è concentrata la distribuzione di popolarità
    def popularity_entropy(pop):
        probs = pop.values
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    entropy_orig = popularity_entropy(pop_original)
    entropy_gen = popularity_entropy(pop_generated)

    # se original > gen (popolarity collapse)
    # se gen > original (distribuzione più uniforme)
    log(f"Popularity entropy (original): {entropy_orig:.4f}")
    log(f"Popularity entropy (generated): {entropy_gen:.4f}")

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

    # gen > original pochi item dominano
    # original > gen distribuzione più piatta
    gini_orig = gini(pop_original.values)
    gini_gen = gini(pop_generated.values)

    log(f"Gini coefficient (original): {gini_orig:.4f}")
    log(f"Gini coefficient (generated): {gini_gen:.4f}")

    # lorenz curve
    def lorenz_curve(values):

        sorted_vals = np.sort(values)
        cumvals = np.cumsum(sorted_vals)

        cumvals = np.insert(cumvals, 0, 0)
        cumvals = cumvals / cumvals[-1]

        x = np.linspace(0,1,len(cumvals))

        return x, cumvals

    x_o, y_o = lorenz_curve(pop_original.values)
    x_g, y_g = lorenz_curve(pop_generated.values)

    plt.plot(x_o, y_o, label="Original")
    plt.plot(x_g, y_g, label="Generated")
    plt.plot([0,1],[0,1], linestyle="--", color="black")

    plt.title("Lorenz Curve")
    plt.legend()

    plt.savefig(os.path.join(output_dir,"lorenz_curve.png"))
    plt.close()

    # Head Overlap
    # quanto coincidono gli item popolari
    def head_overlap(pop_orig, pop_gen, k=50):

        top_orig = set(pop_orig.sort_values(ascending=False).head(k).index)
        top_gen = set(pop_gen.sort_values(ascending=False).head(k).index)

        overlap = len(top_orig.intersection(top_gen)) / k
        return overlap

    overlap_20 = head_overlap(pop_original, pop_generated, 20)
    overlap_50 = head_overlap(pop_original, pop_generated, 50)

    log(f"Head overlap@20: {overlap_20:.4f}")
    log(f"Head overlap@50: {overlap_50:.4f}")

    # Item frequency scatter plot
    # diagonale ok
    # sopra item amplificati
    # sotto item attenuati
    plt.figure(figsize=(6,6))

    plt.scatter(pop_original.values,
                pop_generated.values,
                alpha=0.3)

    max_val = max(pop_original.max(), pop_generated.max())

    plt.plot([0,max_val],[0,max_val], color='red')

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Original frequency")
    plt.ylabel("Generated frequency")
    plt.title("Item frequency scatter")

    plt.savefig(os.path.join(output_dir, "frequency_scatter.png"), dpi=200)
    plt.close()

    # CDF
    cdf_orig = np.cumsum(np.sort(pop_original.values)) / np.sum(pop_original.values)
    cdf_gen = np.cumsum(np.sort(pop_generated.values)) / np.sum(pop_generated.values)


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

    log("\n--- Repeat item distribution (top 10) ---")
    repeat_items = repeats['item'].value_counts(normalize=True)
    log(repeat_items.head(10))

    plt.figure(figsize=(8,4))

    repeat_items.head(20).plot(kind="bar")

    plt.title("Top repeated items")
    plt.ylabel("Fraction of repetitions")

    plt.savefig(os.path.join(output_dir,"repeat_items_barplot.png"))
    plt.close()

    top5_share = repeat_items.head(5).sum()

    log(f"Top-5 items explain {top5_share:.2%} of repetitions")

    # Repetition of Popular Items

    top_pop_items = set(pop_generated.sort_values(ascending=False).head(50).index)
    repeat_top_items = set(repeat_items.head(50).index)

    overlap = len(top_pop_items.intersection(repeat_top_items)) / len(repeat_top_items)

    log(f"Overlap between repeated items and top popularity: {overlap:.2f}")

    repeat_prob = repeat_items.reindex(pop_generated.index).fillna(0)

    pop_rank = pop_generated.rank(ascending=False)

    plt.scatter(pop_rank, repeat_prob, alpha=0.3)

    plt.xlabel("Popularity rank")
    plt.ylabel("Repeat probability")

    plt.savefig(os.path.join(output_dir,"repeat_vs_popularity.png"))

    def count_short_loops(df):

        df = df.copy()

        df['prev_item'] = df.groupby('user')['item'].shift(1)
        df['prev2_item'] = df.groupby('user')['item'].shift(2)

        loops = df[(df['item'] == df['prev2_item'])]

        return len(loops) / len(df)

    loop_orig = count_short_loops(original)
    loop_gen = count_short_loops(generated)

    log(f"Short loops (original): {loop_orig:.4f}")
    log(f"Short loops (generated): {loop_gen:.4f}")

    # 3-cycles

    def count_3_cycles(df):

        df = df.copy()

        df['prev3_item'] = df.groupby('user')['item'].shift(3)

        cycles = df[df['item'] == df['prev3_item']]

        return len(cycles) / len(df)

    cycle3_orig = count_3_cycles(original)
    cycle3_gen = count_3_cycles(generated)

    log(f"3-cycles (original): {cycle3_orig:.4f}")
    log(f"3-cycles (generated): {cycle3_gen:.4f}")

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

    logger.close()





