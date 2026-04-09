from __future__ import annotations

import csv
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class BookRecord:
    asin: str
    filename: str
    image_url: str
    title: str
    author: str
    category_id: str
    category_name: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _iter_csv_rows(csv_path: str | Path) -> Iterable[list[str]]:
    encodings = ("utf-8-sig", "utf-8", "shift_jis", "latin-1")
    for encoding in encodings:
        try:
            with Path(csv_path).open("r", encoding=encoding, errors="ignore", newline="") as handle:
                reader = csv.reader(handle)
                yield from reader
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("bookgenre", b"", 0, 1, "Unable to decode CSV with fallback encodings")


def load_book_records(csv_path: str | Path, max_rows: int | None = None) -> list[BookRecord]:
    records: list[BookRecord] = []
    for row in _iter_csv_rows(csv_path):
        if len(row) < 7:
            continue
        normalized = [cell.strip() for cell in row[:7]]
        if normalized[0].lower().startswith("amazon id"):
            continue
        record = BookRecord(
            asin=normalized[0],
            filename=normalized[1],
            image_url=normalized[2],
            title=normalized[3],
            author=normalized[4],
            category_id=normalized[5],
            category_name=normalized[6],
        )
        records.append(record)
        if max_rows is not None and len(records) >= max_rows:
            break
    return records


def compose_text(record: BookRecord, include_author: bool = True) -> str:
    if include_author and record.author:
        return f"{record.title} [AUTHOR] {record.author}"
    return record.title


def build_label_mapping(records: Iterable[BookRecord]) -> dict[str, int]:
    sortable: list[tuple[int, str]] = []
    for record in records:
        try:
            sortable.append((int(record.category_id), record.category_name))
        except ValueError:
            sortable.append((10**9, record.category_name))
    unique_labels = sorted(set(sortable))
    return {label_name: index for index, (_, label_name) in enumerate(unique_labels)}


def filter_records_with_images(
    records: Iterable[BookRecord], image_root: str | Path
) -> tuple[list[BookRecord], list[BookRecord]]:
    root_path = Path(image_root)
    zip_members: set[str] | None = None
    if root_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(root_path) as archive:
            zip_members = set(archive.namelist())
    kept: list[BookRecord] = []
    missing: list[BookRecord] = []
    for record in records:
        if not _image_exists(record, root_path, zip_members):
            missing.append(record)
        else:
            kept.append(record)
    return kept, missing


def split_records_by_label(
    records: Iterable[BookRecord], validation_ratio: float = 0.1, seed: int = 42
) -> tuple[list[BookRecord], list[BookRecord]]:
    grouped: dict[str, list[BookRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category_name].append(record)

    rng = random.Random(seed)
    train_records: list[BookRecord] = []
    validation_records: list[BookRecord] = []

    for group in grouped.values():
        copied_group = list(group)
        rng.shuffle(copied_group)
        split_index = max(1, int(len(copied_group) * (1 - validation_ratio)))
        split_index = min(split_index, len(copied_group) - 1) if len(copied_group) > 1 else len(copied_group)
        train_records.extend(copied_group[:split_index])
        validation_records.extend(copied_group[split_index:])

    return train_records, validation_records


def balanced_subset(records: Iterable[BookRecord], samples_per_class: int, seed: int = 42) -> list[BookRecord]:
    grouped: dict[str, list[BookRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category_name].append(record)
    rng = random.Random(seed)
    subset: list[BookRecord] = []
    for group in grouped.values():
        copied_group = list(group)
        rng.shuffle(copied_group)
        subset.extend(copied_group[:samples_per_class])
    rng.shuffle(subset)
    return subset


class TextVocabulary:
    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    @classmethod
    def build(
        cls, texts: Iterable[str], max_tokens: int = 20000, min_frequency: int = 1
    ) -> "TextVocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(_tokenize(text))

        token_to_id = {cls.pad_token: 0, cls.unk_token: 1}
        for token, frequency in counter.most_common():
            if frequency < min_frequency:
                continue
            if len(token_to_id) >= max_tokens:
                break
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [self.token_to_id.get(token, 1) for token in _tokenize(text)]
        token_ids = token_ids[:max_length]
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))
        return token_ids

    def save(self, output_path: str | Path) -> None:
        Path(output_path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: str | Path) -> "TextVocabulary":
        token_to_id = json.loads(Path(input_path).read_text(encoding="utf-8"))
        return cls({str(key): int(value) for key, value in token_to_id.items()})

    def __len__(self) -> int:
        return len(self.token_to_id)


class BookMultimodalDataset(Dataset):
    def __init__(
        self,
        records: list[BookRecord],
        image_root: str | Path,
        label_to_index: dict[str, int],
        vocabulary: TextVocabulary,
        max_length: int = 32,
        image_size: int = 224,
        include_author: bool = True,
    ) -> None:
        self.records = records
        self.image_root = Path(image_root)
        self._zip_members: set[str] | None = None
        self.label_to_index = label_to_index
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.include_author = include_author
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = _load_image(record, self.image_root, self._get_zip_members()).convert("RGB")
        image_tensor = self.transform(image)
        text_ids = torch.tensor(
            self.vocabulary.encode(compose_text(record, include_author=self.include_author), self.max_length),
            dtype=torch.long,
        )
        label = self.label_to_index.get(record.category_name, -1)
        return {
            "image": image_tensor,
            "text_ids": text_ids,
            "label": torch.tensor(label, dtype=torch.long),
            "record": record.to_dict(),
        }

    def _get_zip_members(self) -> set[str] | None:
        if self.image_root.suffix.lower() != ".zip":
            return None
        if self._zip_members is None:
            with zipfile.ZipFile(self.image_root) as archive:
                self._zip_members = set(archive.namelist())
        return self._zip_members


def _resolve_image_path(record: BookRecord, image_root: Path) -> Path | None:
    category_path = image_root / record.category_name / record.filename
    flat_path = image_root / record.filename
    if category_path.exists():
        return category_path
    if flat_path.exists():
        return flat_path
    return None


def _candidate_image_members(record: BookRecord) -> list[str]:
    return [
        record.filename,
        f"224x224/{record.filename}",
        f"{record.category_name}/{record.filename}",
    ]


def _image_exists(record: BookRecord, image_root: str | Path, zip_members: set[str] | None = None) -> bool:
    root_path = Path(image_root)
    if root_path.suffix.lower() == ".zip":
        members = zip_members
        if members is None:
            with zipfile.ZipFile(root_path) as archive:
                members = set(archive.namelist())
        return any(member in members for member in _candidate_image_members(record))
    return _resolve_image_path(record, root_path) is not None


def _load_image(record: BookRecord, image_root: Path, zip_members: set[str] | None = None) -> Image.Image:
    if image_root.suffix.lower() == ".zip":
        members = zip_members
        if members is None:
            with zipfile.ZipFile(image_root) as archive:
                members = set(archive.namelist())
        image_member = next((member for member in _candidate_image_members(record) if member in members), None)
        if image_member is None:
            raise FileNotFoundError(f"Image not found for {record.filename} in {image_root}")
        with zipfile.ZipFile(image_root) as archive:
            with archive.open(image_member) as handle:
                return Image.open(io.BytesIO(handle.read())).copy()

    image_path = _resolve_image_path(record, image_root)
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {record.filename}")
    return Image.open(image_path)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]
