import os
import sys
from config import get_config
from dotenv import load_dotenv
from generation import generate_synthetic_data


if __name__ == '__main__':
    args = get_config()

    load_dotenv()
    if not os.path.exists(args.input_file):
        print(f"File {args.input_file} doesn't exists.")
        sys.exit(1)

    if os.path.exists(args.output_file):
        print(f"File {args.output_file} already exists.")
        sys.exit(1)

    print(f"Start data generation with: {args.model_name}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens for generation: {args.max_token}")

    try:
        generate_synthetic_data(args)
    except Exception as e:
        print(f"An error occurred while generating synthetic: {e}")
        sys.exit(1)
