import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from datarec.datasets import load_dataset
from datarec.processing import Binarize, UserItemIterativeKCore


def parse_args():
    parser = argparse.ArgumentParser(description="Reusable DataRec preprocessing for sequential recommendation.")
    parser.add_argument(
        "--dataset",
        default="movielens",
        help="DataRec registry dataset name (e.g. movielens, amazon_beauty, amazon_toys_and_games, yelp).",
    )
    parser.add_argument("--version", default="latest", help="DataRec dataset version.")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent), help="Folder where outputs are saved.")
    parser.add_argument("--output-file", default="interactions.txt", help="Interactions output filename.")
    parser.add_argument("--threshold", type=float, default=4.0, help="Binarization threshold.")
    parser.add_argument("--kcore", type=int, default=5, help="Iterative user/item k-core.")
    return parser.parse_args()


def preprocess_data(data, threshold, kcore):
    data = Binarize(
        threshold=threshold,
        keep="positive",
        drop_rating_col=True,
    ).run(data)
    if kcore > 0:
        data = UserItemIterativeKCore(cores=kcore).run(data)

    return data


def build_interactions(data):
    df = data.data.copy()
    user_col = data.user_col
    item_col = data.item_col
    timestamp_col = data.timestamp_col

    if timestamp_col is None or timestamp_col not in df.columns:
        raise ValueError("Timestamp column is required to sort interactions chronologically.")
    df = df.sort_values(by=[user_col, timestamp_col]).reset_index(drop=True)

    if timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])

    return df[[user_col, item_col]].copy()


def encode_ids(data, interactions):
    user_col = data.user_col
    item_col = data.item_col

    data.user_id_encoder.build_encoding(interactions[user_col].drop_duplicates().tolist(), offset=1)
    data.item_id_encoder.build_encoding(interactions[item_col].drop_duplicates().tolist(), offset=1)

    encoded = interactions.copy()
    encoded[user_col] = data.user_id_encoder.encode(encoded[user_col].tolist())
    encoded[item_col] = data.item_id_encoder.encode(encoded[item_col].tolist())
    encoded = encoded.astype(int)

    id2user = {v: k for k, v in data.user_id_encoder.encoding.items()}
    id2item = {v: k for k, v in data.item_id_encoder.encoding.items()}
    return encoded, id2user, id2item


def save_outputs(output_dir, output_file, encoded, id2user, id2item, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(output_dir / output_file, sep=" ", index=False, header=False)

    with open(output_dir / "user_mapping.pkl", "wb") as f:
        pickle.dump(id2user, f)
    with open(output_dir / "item_mapping.pkl", "wb") as f:
        pickle.dump(id2item, f)
    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_file = args.output_file

    dataset_entrypoint = load_dataset(args.dataset, version=args.version)
    data = dataset_entrypoint.prepare_and_load()

    data = preprocess_data(
        data=data,
        threshold=args.threshold,
        kcore=args.kcore,
    )
    interactions = build_interactions(data=data)
    encoded, id2user, id2item = encode_ids(
        data=data,
        interactions=interactions,
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "version": args.version,
        "threshold": args.threshold,
        "keep_positive_only": True,
        "kcore": args.kcore,
        "sorted_by_timestamp": True,
        "user_offset": 1,
        "item_offset": 1,
        "n_interactions": int(len(encoded)),
        "n_users": int(encoded[data.user_col].nunique()),
        "n_items": int(encoded[data.item_col].nunique()),
        "min_user_id": int(encoded[data.user_col].min()),
        "max_user_id": int(encoded[data.user_col].max()),
        "min_item_id": int(encoded[data.item_col].min()),
        "max_item_id": int(encoded[data.item_col].max()),
        "output_file": output_file,
    }

    save_outputs(
        output_dir=output_dir,
        output_file=output_file,
        encoded=encoded,
        id2user=id2user,
        id2item=id2item,
        metadata=metadata,
    )

    print(f"Saved interactions: {output_dir / output_file}")
    print(f"Saved mappings: {output_dir / 'user_mapping.pkl'}, {output_dir / 'item_mapping.pkl'}")
    print(
        f"Users={metadata['n_users']} [{metadata['min_user_id']}, {metadata['max_user_id']}], "
        f"Items={metadata['n_items']} [{metadata['min_item_id']}, {metadata['max_item_id']}], "
        f"Interactions={metadata['n_interactions']}"
    )


if __name__ == "__main__":
    main()
