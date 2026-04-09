from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bookgenre.checkpoints import save_checkpoint_bundle
from bookgenre.data import (
    BookMultimodalDataset,
    TextVocabulary,
    balanced_subset,
    build_label_mapping,
    compose_text,
    filter_records_with_images,
    load_book_records,
    split_records_by_label,
)
from bookgenre.model import MultimodalGenreModel
from bookgenre.training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal book genre classifier.")
    parser.add_argument("--train-csv", required=True, help="Training manifest CSV path.")
    parser.add_argument("--valid-csv", help="Validation manifest CSV path.")
    parser.add_argument("--image-root", required=True, help="Root directory for downloaded images.")
    parser.add_argument("--output-dir", required=True, help="Directory for model outputs.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap on training rows.")
    parser.add_argument("--max-valid-rows", type=int, default=None, help="Optional cap on validation rows.")
    parser.add_argument("--balanced-per-class", type=int, default=None, help="Optional balanced sampling cap.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Used when --valid-csv is omitted.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-patience", type=int, default=1)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--text-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--image-only", action="store_true", help="Disable title/author text fusion.")
    parser.add_argument("--include-author", action="store_true", help="Include author text in the text field.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run even if a matching checkpoint exists.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    train_records = load_book_records(args.train_csv, max_rows=args.max_train_rows)
    if args.balanced_per_class is not None:
        train_records = balanced_subset(train_records, args.balanced_per_class)

    if args.valid_csv:
        valid_records = load_book_records(args.valid_csv, max_rows=args.max_valid_rows)
    else:
        train_records, valid_records = split_records_by_label(
            train_records, validation_ratio=args.validation_ratio
        )

    train_records, missing_train = filter_records_with_images(train_records, args.image_root)
    valid_records, missing_valid = filter_records_with_images(valid_records, args.image_root)

    label_to_index = build_label_mapping(train_records + valid_records)
    texts = [compose_text(record, include_author=args.include_author) for record in train_records]
    vocabulary = TextVocabulary.build(texts, max_tokens=args.max_vocab)

    train_dataset = BookMultimodalDataset(
        records=train_records,
        image_root=args.image_root,
        label_to_index=label_to_index,
        vocabulary=vocabulary,
        max_length=args.max_length,
        include_author=args.include_author,
    )
    valid_dataset = BookMultimodalDataset(
        records=valid_records,
        image_root=args.image_root,
        label_to_index=label_to_index,
        vocabulary=vocabulary,
        max_length=args.max_length,
        include_author=args.include_author,
    )

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = MultimodalGenreModel(
        num_classes=len(label_to_index),
        vocab_size=len(vocabulary),
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        text_hidden_dim=args.text_hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        use_text=not args.image_only,
    )

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        device=args.device,
        use_amp=args.amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        resume_if_possible=not args.no_resume,
    )

    output_dir = Path(args.output_dir)
    run_signature = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "valid_csv": str(Path(args.valid_csv).resolve()) if args.valid_csv else None,
        "image_root": str(Path(args.image_root).resolve()),
        "balanced_per_class": args.balanced_per_class,
        "validation_ratio": args.validation_ratio if not args.valid_csv else None,
        "max_vocab": args.max_vocab,
        "max_length": args.max_length,
        "text_hidden_dim": args.text_hidden_dim,
        "projection_dim": args.projection_dim,
        "dropout": args.dropout,
        "backbone": args.backbone,
        "pretrained_backbone": args.pretrained_backbone,
        "use_text": not args.image_only,
        "include_author": args.include_author,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
    }
    summary = train_model(
        model,
        train_loader,
        valid_loader,
        config=config,
        output_dir=output_dir,
        run_signature=run_signature,
    )

    metadata = {
        "num_classes": len(label_to_index),
        "vocab_size": len(vocabulary),
        "backbone_name": args.backbone,
        "text_hidden_dim": args.text_hidden_dim,
        "projection_dim": args.projection_dim,
        "dropout": args.dropout,
        "use_text": not args.image_only,
        "max_length": args.max_length,
        "include_author": args.include_author,
        "missing_train_images": len(missing_train),
        "missing_valid_images": len(missing_valid),
        "train_examples": len(train_dataset),
        "validation_examples": len(valid_dataset),
        "device": args.device,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "best_epoch": summary.get("best_epoch"),
        "requested_epochs": args.epochs,
    }
    save_checkpoint_bundle(output_dir, model, label_to_index, vocabulary, metadata)
    (output_dir / "data_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved checkpoint bundle to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
