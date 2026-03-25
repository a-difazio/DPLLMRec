import os
import json
import pandas as pd
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SASREC_DIR = os.path.dirname(SCRIPT_DIR)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    return vars(parser.parse_args())

def load_all_stats(stats_dir):
    rows = []

    for dirpath, _, filenames in os.walk(stats_dir):
        for file in filenames:
            if file.startswith("stats_") and file.endswith(".json"):
                filepath = os.path.join(dirpath, file)

                with open(filepath) as f:
                    stats = json.load(f)

                stats['config'] = file.replace("stats_", "").replace(".json", "")
                rows.append(stats)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    args = parse()
    experiment_name = f"{args['dataset']}_{args['run_name']}"
    stats_dir = os.path.join(SASREC_DIR, "experiments", experiment_name, "generated_stats")
    output_file = os.path.join(stats_dir, "aggregate_stats.csv")

    stats = load_all_stats(stats_dir)
    stats = stats.sort_values(by=['config']).reset_index(drop=True)
    stats.to_csv(output_file, index=False)

    print(f"Wrote aggregated stats to {output_file}.")