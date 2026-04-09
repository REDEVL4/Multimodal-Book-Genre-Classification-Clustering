from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bookgenre.checkpoints import load_checkpoint_bundle
from bookgenre.data import BookMultimodalDataset, filter_records_with_images, load_book_records
from bookgenre.inference import predict_dataset, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confidence-aware genre inference on a catalog manifest.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--review-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    model, label_to_index, vocabulary, metadata = load_checkpoint_bundle(args.checkpoint_dir, device=args.device)
    records = load_book_records(args.manifest_csv)
    records, missing_records = filter_records_with_images(records, args.image_root)

    dataset = BookMultimodalDataset(
        records=records,
        image_root=args.image_root,
        label_to_index=label_to_index,
        vocabulary=vocabulary,
        max_length=int(metadata["max_length"]),
        include_author=bool(metadata["include_author"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    predictions = predict_dataset(
        model=model,
        loader=loader,
        label_to_index=label_to_index,
        device=args.device,
        top_k=args.top_k,
        review_threshold=args.review_threshold,
    )
    write_jsonl(predictions, args.output_file)
    print(f"Wrote {len(predictions)} predictions to {args.output_file}")
    print(f"Skipped {len(missing_records)} records with missing images")


if __name__ == "__main__":
    main()
