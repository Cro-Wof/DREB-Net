#!/usr/bin/env python3
"""Draw VisDrone frame-level detection annotations on generated images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


CATEGORY_REMAP = {
    1: ("people", (0, 255, 0)),
    2: ("people", (0, 255, 0)),
    4: ("car", (0, 165, 255)),
    5: ("car", (0, 165, 255)),
    6: ("truck", (255, 0, 0)),
    9: ("bus", (255, 0, 255)),
}
RAW_CATEGORY_STYLE = {
    0: ("ignored_region", (128, 128, 128)),
    1: ("pedestrian", (0, 255, 0)),
    2: ("people", (0, 255, 0)),
    3: ("bicycle", (255, 255, 0)),
    4: ("car", (0, 165, 255)),
    5: ("van", (0, 165, 255)),
    6: ("truck", (255, 0, 0)),
    7: ("tricycle", (0, 255, 255)),
    8: ("awning_tricycle", (0, 128, 255)),
    9: ("bus", (255, 0, 255)),
    10: ("motor", (0, 0, 255)),
    11: ("others", (128, 0, 128)),
}


def read_labels(path: Path, include_non_dreb: bool = False):
    labels = []
    if not path.exists():
        raise FileNotFoundError(f"missing annotation file: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.rstrip(",").split(",")
        if len(fields) != 8:
            raise ValueError(
                f"{path}:{line_number}: expected 8 fields, got {len(fields)}"
            )
        x, y, width, height, score, category, truncation, occlusion = [
            int(value) for value in fields
        ]
        if category in CATEGORY_REMAP:
            labels.append((x, y, width, height, CATEGORY_REMAP[category]))
        elif include_non_dreb and category in RAW_CATEGORY_STYLE:
            labels.append((x, y, width, height, RAW_CATEGORY_STYLE[category]))
    return labels


def draw_frame(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    include_non_dreb: bool = False,
) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {image_path}")
    height, width = image.shape[:2]
    thickness = max(1, min(width, height) // 600)
    font_scale = max(0.4, min(width, height) / 1800.0)
    labels = read_labels(label_path, include_non_dreb=include_non_dreb)

    for x, y, box_width, box_height, (name, color) in labels:
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width - 1, x + box_width))
        y2 = max(0, min(height - 1, y + box_height))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text_y = max(15, y1 - 4)
        cv2.putText(
            image,
            name,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]):
        raise RuntimeError(f"failed to write visualization: {output_path}")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-non-dreb",
        action="store_true",
        help="Also draw raw VisDrone categories excluded from the four DREB classes.",
    )
    args = parser.parse_args()

    image_paths = sorted(
        args.image_dir.glob("*.jpg"), key=lambda path: int(path.stem)
    )
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise RuntimeError(f"no JPG images found in {args.image_dir}")

    total_boxes = 0
    for image_path in image_paths:
        label_path = args.label_dir / f"{image_path.stem}.txt"
        output_path = args.output_dir / image_path.name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output already exists: {output_path}; use --overwrite"
            )
        count = draw_frame(
            image_path,
            label_path,
            output_path,
            include_non_dreb=args.include_non_dreb,
        )
        total_boxes += count
        print(f"{image_path.name}: {count} boxes")

    print(
        f"visualized {len(image_paths)} images, {total_boxes} valid detection boxes; "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
