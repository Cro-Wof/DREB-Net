#!/usr/bin/env python3
"""Prepare VisDrone-VID annotations for the single-frame DREB pipeline.

The raw VisDrone-VID annotations are stored one file per sequence, while the
DREB loader consumes one COCO image record per image.  This script therefore
does three things for each split:

1. Writes one DET-style annotation file per video frame.
2. Builds a COCO JSON containing all frames, including frames without objects.
3. Writes a manifest linking the source sharp frame, future blur frame, and
   frame annotation.

The source dataset is never modified.  Generated files are written below
``--output-root``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import cv2


SPLITS = ("train", "val", "test-dev")

# Raw VisDrone category -> DREB category.  DREB uses category ids 0..3.
CATEGORY_REMAP = {
    1: 0,  # pedestrian
    2: 0,  # people
    4: 1,  # car
    5: 1,  # van
    6: 2,  # truck
    9: 3,  # bus
}
IGNORE_CATEGORIES = {0, 3, 7, 8, 10, 11}
CATEGORIES = [
    {"id": 0, "name": "people", "supercategory": "none"},
    {"id": 1, "name": "car", "supercategory": "none"},
    {"id": 2, "name": "truck", "supercategory": "none"},
    {"id": 3, "name": "bus", "supercategory": "none"},
]


class Record(NamedTuple):
    frame_id: int
    target_id: int
    x: int
    y: int
    width: int
    height: int
    score: int
    category: int
    truncation: int
    occlusion: int


def parse_record(line: str, path: Path, line_number: int) -> Record:
    fields = line.strip().rstrip(",").split(",")
    if len(fields) != 10:
        raise ValueError(
            f"{path}:{line_number}: expected 10 comma-separated fields, "
            f"got {len(fields)}: {line.strip()!r}"
        )

    try:
        values = [int(value.strip()) for value in fields]
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: all annotation fields must be integers"
        ) from exc

    record = Record(*values)
    if record.frame_id < 1:
        raise ValueError(f"{path}:{line_number}: invalid frame id {record.frame_id}")
    if record.width < 0 or record.height < 0:
        raise ValueError(f"{path}:{line_number}: negative bbox size")
    if record.category < 0 or record.category > 11:
        raise ValueError(
            f"{path}:{line_number}: unexpected VisDrone category {record.category}"
        )
    return record


def read_sequence_annotations(path: Path) -> Dict[int, List[Record]]:
    grouped: Dict[int, List[Record]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip().strip(","):
                continue
            record = parse_record(line, path, line_number)
            grouped[record.frame_id].append(record)
    return dict(grouped)


def image_files(sequence_dir: Path) -> List[Path]:
    files = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(files, key=lambda path: int(path.stem))


def check_image(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape for {path}: {image.shape}")
    return width, height


def write_frame_annotation(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    count = 0
    for record in records:
        # Keep the raw category here.  Category remapping is applied while
        # constructing COCO so the frame-level files remain auditable.
        lines.append(
            ",".join(
                str(value)
                for value in (
                    record.x,
                    record.y,
                    record.width,
                    record.height,
                    record.score,
                    record.category,
                    record.truncation,
                    record.occlusion,
                )
            )
        )
        count += 1
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return count


def build_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    check_images: bool = True,
) -> dict:
    source_split_root = dataset_root / f"VisDrone2019-VID-{split}"
    sequences_root = source_split_root / "sequences"
    raw_annotations_root = source_split_root / "annotations"
    split_output_root = output_root / split
    frame_annotations_root = split_output_root / "annotations_frame"

    if not sequences_root.is_dir():
        raise FileNotFoundError(f"missing sequence directory: {sequences_root}")
    if not raw_annotations_root.is_dir():
        raise FileNotFoundError(f"missing annotation directory: {raw_annotations_root}")

    sequences = sorted(path for path in sequences_root.iterdir() if path.is_dir())
    if not sequences:
        raise ValueError(f"no sequence directories found in {sequences_root}")

    coco = {
        "info": {
            "description": "VisDrone-VID prepared for DREB",
            "source_split": split,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    manifest_rows = []
    image_id = 0
    annotation_id = 0
    summary = {
        "split": split,
        "sequences": len(sequences),
        "frames": 0,
        "raw_records": 0,
        "valid_dreb_records": 0,
        "ignored_records": 0,
        "unknown_records": 0,
        "empty_frames": 0,
        "boundary_frames": 0,
        "max_valid_objects_per_frame": 0,
    }

    for sequence_dir in sequences:
        sequence_id = sequence_dir.name
        raw_annotation_path = raw_annotations_root / f"{sequence_id}.txt"
        if not raw_annotation_path.is_file():
            raise FileNotFoundError(f"missing annotation file: {raw_annotation_path}")

        grouped = read_sequence_annotations(raw_annotation_path)
        frames = image_files(sequence_dir)
        if not frames:
            raise ValueError(f"no image frames found in {sequence_dir}")

        frame_ids = {int(path.stem) for path in frames}
        annotation_frame_ids = set(grouped)
        extra_frame_ids = annotation_frame_ids - frame_ids
        if extra_frame_ids:
            examples = sorted(extra_frame_ids)[:5]
            raise ValueError(
                f"{raw_annotation_path}: annotation frame ids do not exist as images; "
                f"examples={examples}"
            )

        sequence_frame_annotation_root = frame_annotations_root / sequence_id
        for frame_position, frame_path in enumerate(frames):
            frame_id = int(frame_path.stem)
            records = grouped.get(frame_id, [])
            valid_records = []

            for record in records:
                summary["raw_records"] += 1
                if record.category in CATEGORY_REMAP:
                    valid_records.append(record)
                elif record.category in IGNORE_CATEGORIES:
                    summary["ignored_records"] += 1
                else:
                    summary["unknown_records"] += 1

            if check_images:
                width, height = check_image(frame_path)
            else:
                width, height = 0, 0

            frame_annotation_path = sequence_frame_annotation_root / f"{frame_path.stem}.txt"
            write_frame_annotation(frame_annotation_path, records)

            file_name = f"{sequence_id}/{frame_path.name}"
            coco["images"].append(
                {
                    "id": image_id,
                    "license": 1,
                    "height": height,
                    "width": width,
                    "file_name": file_name,
                }
            )

            for record in valid_records:
                category_id = CATEGORY_REMAP[record.category]
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [record.x, record.y, record.width, record.height],
                        "area": record.width * record.height,
                        "iscrowd": 0,
                        "ignore": 0,
                        "track_id": record.target_id,
                        "score": record.score,
                        "truncation": record.truncation,
                        "occlusion": record.occlusion,
                    }
                )
                annotation_id += 1

            relative_source_path = (
                Path(f"VisDrone2019-VID-{split}") / "sequences" / file_name
            )
            relative_blur_path = Path(split) / "blur_images" / file_name
            relative_label_path = (
                Path(split) / "annotations_frame" / sequence_id / f"{frame_path.stem}.txt"
            )
            manifest_rows.append(
                {
                    "blur_path": relative_blur_path.as_posix(),
                    "sharp_path": relative_source_path.as_posix(),
                    "label_path": relative_label_path.as_posix(),
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "boundary_flag": int(frame_position in {0, len(frames) - 1}),
                }
            )

            summary["frames"] += 1
            summary["valid_dreb_records"] += len(valid_records)
            summary["max_valid_objects_per_frame"] = max(
                summary["max_valid_objects_per_frame"], len(valid_records)
            )
            if not records:
                summary["empty_frames"] += 1
            if frame_position in {0, len(frames) - 1}:
                summary["boundary_frames"] += 1
            image_id += 1

    split_output_root.mkdir(parents=True, exist_ok=True)
    coco_path = split_output_root / "annotations_dreb4.json"
    manifest_path = split_output_root / "manifest.csv"
    summary_path = split_output_root / "annotation_summary.json"

    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "blur_path",
                "sharp_path",
                "label_path",
                "sequence_id",
                "frame_id",
                "boundary_flag",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing VisDrone2019-VID-train/val/test-dev.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where prepared annotations and manifests are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to process; test-challenge has no public annotations.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip decoding every source image when generating COCO dimensions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing generated files under output-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_root}; use --overwrite explicitly"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for split in args.splits:
        summary = build_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            check_images=not args.skip_image_check,
        )
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    overall = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": all_summaries,
    }
    (output_root / "prepare_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(f"Prepared annotations written to {output_root}")


if __name__ == "__main__":
    main()
