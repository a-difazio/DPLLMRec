import os
from tqdm import tqdm
from types import SimpleNamespace

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

def generate_sequences(user_sequences, itemnum, model, config, options):
    context_len = options.get("context_len", 5)
    max_gen_len = options.get("max_gen_len", None)
    temperature = options.get("temperature", 1.0)
    penalty = options.get("penalty", 0.0)
    no_repeat = options.get("no_repeat", False)
    include_context = options.get("include_context", True)
    top_k = options.get("top_k", None)
    top_p = options.get("top_p", None)

    synthetic = {}

    for user, seq in tqdm(user_sequences.items()):
        generated = seq[:context_len]
        gen_length = max_gen_len or (len(seq) - context_len)

        seq_tensor = torch.zeros(config.maxlen, dtype=torch.long, device=config.device)
        counts = torch.zeros(itemnum+1, dtype=torch.long, device=config.device)

        for _ in range(gen_length):
            input_seq = generated[-config.maxlen:]
            seq_tensor[:] = 0
            seq_tensor[-len(input_seq):] = torch.tensor(input_seq, dtype=torch.long, device=config.device)
            seq_tensor_input = seq_tensor.unsqueeze(0)

            logits = model.predict(user, seq_tensor_input) / temperature
            # masking sullo zero
            logits[:, 0] = float('-inf')

            # Anti-repetition
            if no_repeat:
                logits[generated] = float('-inf')

            probs = logits.softmax(dim=-1)

            # Penalità anti-repetition soft
            if not no_repeat and penalty > 0:
                probs *= penalty ** counts.float()

            # Top-p / Top-k sampling
            if top_p is not None:  # Priorità top-p
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= top_p
                mask[0] = True  # almeno un item
                probs_filtered = torch.zeros_like(probs)
                probs_filtered[sorted_idx[mask]] = probs[sorted_idx[mask]]
                probs = probs_filtered
            elif top_k is not None:
                topk_vals, topk_idx = torch.topk(probs, k=top_k)
                mask = torch.zeros_like(probs)
                mask[topk_idx] = 1
                probs = probs * mask

            # Normalizzazione finale
            probs = probs / probs.sum()

            next_item = torch.multinomial(probs, 1).item()
            generated.append(next_item)
            counts[next_item] += 1
            assert counts[0] == 0, "Padding generated as next item!"

        synthetic[user] = generated if include_context else generated[context_len:]

    return synthetic

def save_synthetic(results_dir, dataset, options, synthetic):
    filename = f"synthetic_{dataset}_temp{options['temperature']}_pen{options['penalty']}_norep{options['no_repeat']}.txt"
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
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    # Generation
    experiments = {
        # Baseline soft
        "baseline_free": {
            "context_len": 5, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": False,
            "include_context": False, "top_k": None, "top_p": None
        },
        "baseline_deterministic": {
            "context_len": 5, "temperature": 0.5,
            "penalty": 0.0, "no_repeat": False,
            "include_context": False, "top_k": None, "top_p": None
        },
        "baseline_creative": {
            "context_len": 5, "temperature": 1.5,
            "penalty": 0.0, "no_repeat": False,
            "include_context": False, "top_k": None, "top_p": None
        },

        # Soft anti-repetition con top-k
        "soft_anti_repetition_top_k": {
            "context_len": 5, "temperature": 1.0,
            "penalty": 0.8, "no_repeat": False,
            "include_context": False, "top_k": 10, "top_p": None
        },

        # Hard no-repetition con top-p
        "hard_no_repetition_top_p": {
            "context_len": 5, "temperature": 1.0,
            "penalty": 0.0, "no_repeat": True,
            "include_context": False, "top_k": None, "top_p": 0.9
        },

        # Hard no-repetition e creativa con top-p
        "hard_no_repetition_creative_top_p": {
            "context_len": 5, "temperature": 1.5,
            "penalty": 0.0, "no_repeat": True,
            "include_context": False, "top_k": None, "top_p": 0.9
        }
    }

    for name, opts in experiments.items():
        print(f"Running generation with options: {opts}")
        with torch.no_grad():
            synthetic = generate_sequences(user_sequences, itemnum, model, config, opts)
        save_synthetic(results_dir, args.dataset, opts, synthetic)
