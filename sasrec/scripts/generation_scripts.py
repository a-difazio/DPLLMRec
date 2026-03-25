# generate_scripts.py

DATASETS = {
    'movielens': 'final',
    'amazon_music': 'maxlen50_final',
    'amazon_games': 'maxlen50_final',
    'amazon_cds': 'maxlen50_final',
}

CHECKPOINT = 'checkpoint_best.pth'
SEED = 42

# Griglia esperimenti
BLOCKS = [
    # Blocco 1 — solo temperatura
    {'temperature': 0.5,  'top_p': None, 'top_k': None, 'penalty': 0.0},
    {'temperature': 0.7,  'top_p': None, 'top_k': None, 'penalty': 0.0},
    {'temperature': 1.0,  'top_p': None, 'top_k': None, 'penalty': 0.0},
    {'temperature': 1.2,  'top_p': None, 'top_k': None, 'penalty': 0.0},
    # Blocco 2 — solo top-p
    {'temperature': 1.0,  'top_p': 0.9,  'top_k': None, 'penalty': 0.0},
    {'temperature': 1.0,  'top_p': 0.95, 'top_k': None, 'penalty': 0.0},
    # Blocco 3 — solo top-k
    {'temperature': 1.0,  'top_p': None, 'top_k': 10,   'penalty': 0.0},
    {'temperature': 1.0,  'top_p': None, 'top_k': 50,   'penalty': 0.0},
    # Blocco 4 — solo penalty
    {'temperature': 1.0,  'top_p': None, 'top_k': None, 'penalty': 0.1},
    {'temperature': 1.0,  'top_p': None, 'top_k': None, 'penalty': 0.3},
]

def make_gen_name(params):
    top_p = params['top_p'] if params['top_p'] is not None else 'none'
    top_k = params['top_k'] if params['top_k'] is not None else 'none'
    return (f"temp{params['temperature']}_"
            f"topp{top_p}_"
            f"topk{top_k}_"
            f"pen{params['penalty']}")


def make_command(dataset, train_run, params):
    gen_name = make_gen_name(params)

    cmd = (f"python generate.py \\\n"
           f"    --dataset {dataset} \\\n"
           f"    --run_name {train_run} \\\n"
           f"    --gen_name {gen_name} \\\n"
           f"    --checkpoint {CHECKPOINT} \\\n"
           f"    --device $DEVICE \\\n"
           f"    --temperature {params['temperature']} \\\n"
           f"    --penalty {params['penalty']} \\\n"
           f"    --seed {SEED}")

    if params['top_p'] is not None:
        cmd += f" \\\n    --top_p {params['top_p']}"
    if params['top_k'] is not None:
        cmd += f" \\\n    --top_k {params['top_k']}"

    return cmd


def make_script(dataset, train_run):
    lines = [
        f"#!/bin/bash",
        f"# run_generate_{dataset}.sh",
        f"",
        f"DEVICE=${{1:-cuda}}  # default cuda, override con: bash script.sh mps",
        f"",
        f"echo '=== Generating for {dataset} (device: $DEVICE) ==='",
        f"",
    ]

    for params in BLOCKS:
        gen_name = make_gen_name(params)
        lines.append(f"echo 'Running {gen_name}'")
        lines.append(make_command(dataset, train_run, params))
        lines.append("")

    lines.append(f"echo '=== Done {dataset} ==='")
    return "\n".join(lines)


if __name__ == '__main__':
    for dataset, train_run in DATASETS.items():
        script = make_script(dataset, train_run)
        filename = f"run_generate_{dataset}.sh"
        with open(filename, 'w') as f:
            f.write(script)
        print(f"Generated {filename} ({len(BLOCKS)} experiments)")

    print(f"\nTotal experiments: {len(DATASETS) * len(BLOCKS)}")