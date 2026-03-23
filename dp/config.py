import argparse


def get_config():
    parser = argparse.ArgumentParser(description="Synthetic text dataset generator")

    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_private_token", type=int, default=50)
    parser.add_argument("--tau_private", type=float, default=1.0)
    parser.add_argument("--tau_public", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--public_prompt", type=str, default="Generate a short phrase.")
    parser.add_argument("--input_file", type=str, default="prompt.txt")
    parser.add_argument("--output_file", type=str, default="synthetic.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clipping_bound", type=int, default=50)

    return parser.parse_args()
