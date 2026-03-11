import os
from tqdm import tqdm
from types import SimpleNamespace
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

def save_synthetic(results_dir, dataset, options, synthetic):
    run_tag = options.get("name", "run")
    filename = f"synthetic_{dataset}_{run_tag}.txt"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        for user_id, items in synthetic.items():
            for item in items:
                f.write(f"{user_id} {item}\n")

    print(f'Results saved to {filepath}.')


if __name__ == '__main__':

    # Parser
    args = parser()

    # Setup Seed and Artifacts
    set_seed(args.seed)
    checkpoint_path, config_path, results_dir = setup_paths(args)
    config = load_config(config_path)
    config = {**config, "device": args.device}
    config = SimpleNamespace(**config)
    checkpoint = load_checkpoint(checkpoint_path, args.device)
    user_sequences, usernum, itemnum = load_interactions(os.path.join('data', f"{args.dataset}.txt"))

    # Model Instantiation
    model = SASRec(usernum, itemnum, config)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    # Generation
    experiments = {
        # 1) Baseline puro: nessun controllo oltre temperature
        "greedy_like_temp_0_7": {
            "context_len": 5, "max_gen_len": None, "temperature": 0.7,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": None
        },
        "baseline_temp_1_0": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": None
        },
        "creative_temp_1_3": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.3,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": None
        },

        # 2) Solo top-k (nucleus spento)
        "topk_10_temp_1_0": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": 10, "top_p": None
        },
        "topk_50_temp_1_0": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": 50, "top_p": None
        },

        # 3) Solo top-p (top-k spento)
        "topp_0_9_temp_1_0": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": 0.9
        },
        "topp_0_95_temp_1_0": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": 0.95
        },

        # 4) Solo controllo ripetizione
        "softrep_pen_0_1": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.1, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": None
        },
        "softrep_pen_0_3": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.3, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": None
        },
        "hard_norepeat": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": True, "include_context": False,
            "top_k": None, "top_p": None
        },

        # 5) Combinazioni pragmatiche
        "best_guess_topk10_softrep": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.1, "no_repeat": False, "include_context": False,
            "top_k": 10, "top_p": None
        },
        "best_guess_topp0_9_softrep": {
            "context_len": 5, "max_gen_len": None, "temperature": 1.0,
            "penalty": 0.1, "no_repeat": False, "include_context": False,
            "top_k": None, "top_p": 0.9
        },
        "conservative_topp0_9_hard": {
            "context_len": 5, "max_gen_len": None, "temperature": 0.9,
            "penalty": 0.0, "no_repeat": True, "include_context": False,
            "top_k": None, "top_p": 0.9
        },
    }

    for name, opts in experiments.items():
        print(f"Running generation with options: {opts}")
        synthetic = {}
        for user, sequence in tqdm(user_sequences.items()):
            synthetic[user] = model.generate(user, sequence, config, opts)

        save_synthetic(results_dir, args.dataset, opts, synthetic)
