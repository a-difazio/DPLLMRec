import os
from types import SimpleNamespace
import time
import torch

import argparse
import json

from model import SASRec
from utils import *


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='mps', type=str)
    parser.add_argument('--checkpoint', default='checkpoint_best.pth', type=str)
    # parametri di generazione
    parser.add_argument('--gen_name', required=True)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=None, type=float)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--penalty', default=0.0, type=float)
    parser.add_argument('--no_repeat', action='store_true', default=False)
    parser.add_argument('--context_len', default=5, type=int)
    parser.add_argument('--include_context', action='store_true', default=False)
    return parser.parse_args()

def setup_paths(args):
    base_dir = os.path.join('experiments', f'{args.dataset}_{args.run_name}')
    checkpoint_path = os.path.join(base_dir, 'checkpoints', args.checkpoint)
    config_path = os.path.join(base_dir, 'config.json')
    results_dir = os.path.join(base_dir, 'generated_data')

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    os.makedirs(results_dir, exist_ok=True)
    print(f'Results directory {results_dir} created.')

    return checkpoint_path, config_path, results_dir

def load_config(path):

    with open(path, 'r') as f:
        config = json.load(f)

    print(f'Config loaded from {path}.')

    return config

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)

    print(f'Checkpoint loaded from {path}.')

    return checkpoint

def save_config(output_path, args):
    path = os.path.join(output_path, 'config.json')

    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'Config saved to {path}.')


def save_synthetic(filepath, synthetic):

    with open(filepath, "w") as f:
        for user_id, items in synthetic.items():
            for item in items:
                f.write(f"{user_id}\t{item}\n")

    print(f'Results saved to {filepath}.')


if __name__ == '__main__':

    # Parser
    args = parser()

    # Setup Seed and Artifacts
    set_seed(args.seed)
    checkpoint_path, config_path, results_dir = setup_paths(args)
    model_config = load_config(config_path)
    model_config = {**model_config, "device": args.device}
    model_config = SimpleNamespace(**model_config)
    checkpoint = load_checkpoint(checkpoint_path, args.device)
    user_sequences, usernum, itemnum = load_interactions(os.path.join('data', f"{args.dataset}.tsv"))

    # Model Instantiation
    model = SASRec(usernum, itemnum, model_config)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()

    start_time = time.time()

    # Generation
    print(f"Running generation: {args.run_name}")
    synthetic = {}
    users = list(user_sequences.items())
    total = len(users)
    for i, (user, sequence) in enumerate(users):
        synthetic[user] = model.generate(user, sequence, model_config, args)
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Generated {i+1}/{total} users | elapsed: {elapsed:.1f}s", flush=True)

    output_file = os.path.join(results_dir,f"synthetic_{args.dataset}_{args.gen_name}.tsv")
    save_config(results_dir, args)
    save_synthetic(output_file, synthetic)

    total_time = time.time() - start_time
    print(f"Generation complete in {total_time:.1f}s "
          f"({total_time / total * 1000:.1f}ms per user)", flush=True)
