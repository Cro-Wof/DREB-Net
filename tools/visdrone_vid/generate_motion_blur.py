#!/usr/bin/env python3
"""Generate motion-consistent blur for VisDrone-VID sequences.

For each center frame I[t], this script creates virtual sharp subframes by
warping I[t] along the optical flow to its adjacent frame(s), then integrates
the subframes over a short exposure interval:

    B[t] = sum_k w[k] * S[t, k],    sum_k w[k] = 1

The implementation processes one sequence as a stream.  It keeps only three
images and the optical flow for adjacent pairs in memory, so it does not cache
all video frames or all sequence-level flow fields.

The RIFE backend is the recommended interpolation path.  It uses actual
adjacent source frames and arbitrary-timestep RIFE_m interpolation for every
exposure sample.  The Farneback backend remains available as a self-contained
fallback.  For exposure spans above two frames, Farneback extrapolates the
nearest adjacent flow and should be treated as an exploratory setting rather
than a physically complete multi-frame model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


SPLITS = ("train", "val", "test-dev")


def numeric_image_files(sequence_dir: Path) -> List[Path]:
    files = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(files, key=lambda path: int(path.stem))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"expected a 3-channel image: {path}, shape={image.shape}")
    return image


def check_same_shape(first: np.ndarray, second: np.ndarray, first_path: Path, second_path: Path) -> None:
    if first.shape != second.shape:
        raise RuntimeError(
            f"all frames in a sequence must have the same shape; "
            f"{first_path}={first.shape}, {second_path}={second.shape}"
        )


def estimate_farneback_pair(
    first: np.ndarray,
    second: np.ndarray,
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return flow first->second and flow second->first.

    OpenCV's flow convention is displacement in the source image coordinate
    system: destination ~= source + flow[source].
    """

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    flags = 0
    forward = cv2.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    backward = cv2.calcOpticalFlowFarneback(
        second_gray,
        first_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    return forward, backward


def warp_by_fraction(
    image: np.ndarray,
    flow: np.ndarray,
    fraction: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Warp an image by a fractional source-to-neighbor displacement.

    This is a short-exposure approximation: for a destination pixel x, sample
    the center image at x - fraction * flow(x).  The sign is important because
    cv2.remap uses a backward sampling map while flow is a forward displacement.
    """

    map_x = grid_x - np.float32(fraction) * flow[..., 0]
    map_y = grid_y - np.float32(fraction) * flow[..., 1]
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_offsets_and_weights(
    exposure_span: float,
    num_subframes: int,
    weight_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if exposure_span <= 0:
        raise ValueError("exposure-span must be positive")
    if num_subframes < 3 or num_subframes % 2 == 0:
        raise ValueError("num-subframes must be an odd integer >= 3")

    offsets = np.linspace(
        -exposure_span / 2.0,
        exposure_span / 2.0,
        num_subframes,
        dtype=np.float32,
    )
    if weight_mode == "uniform":
        weights = np.ones(num_subframes, dtype=np.float32)
    elif weight_mode == "triangular":
        center = (num_subframes - 1) / 2.0
        weights = 1.0 - np.abs(np.arange(num_subframes, dtype=np.float32) - center) / (center + 1.0)
    else:
        raise ValueError(f"unsupported weight mode: {weight_mode}")
    weights /= weights.sum()
    return offsets, weights


class RIFEInterpolator:
    """Thin inference wrapper around the official PyTorch RIFE repository.

    This uses ``Model(arbitrary=True)`` from RIFE's IFNet_m implementation so
    each exposure sample can request its actual timestep between two adjacent
    source frames.  The model directory must contain the official RIFE_m
    ``flownet.pkl`` checkpoint.
    """

    def __init__(
        self,
        rife_root: Path,
        model_dir: Path,
        device_name: str,
        scale: float,
        fp16: bool,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError(
                "RIFE backend requires PyTorch; install a supported torch/torchvision pair"
            ) from exc

        self.torch = torch
        self.F = F
        if not rife_root.is_dir():
            raise FileNotFoundError(f"RIFE repository not found: {rife_root}")
        if not (rife_root / "model").is_dir():
            raise FileNotFoundError(f"RIFE model package not found under: {rife_root}")
        if not (model_dir / "flownet.pkl").is_file():
            raise FileNotFoundError(
                f"RIFE checkpoint not found: {model_dir / 'flownet.pkl'}"
            )

        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable. Check the NVIDIA driver "
                    "and install a CUDA-matched PyTorch build in DREBNet."
                )
            self.device = torch.device("cuda")
        elif device_name == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scale = float(scale)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        sys.path.insert(0, str(rife_root))
        try:
            import model.RIFE as rife_module
        except ImportError as exc:
            raise RuntimeError(
                "cannot import RIFE. Make sure --rife-root points to ECCV2022-RIFE"
            ) from exc

        # The upstream module keeps its device in a module-level variable.
        # Override it so --device cpu remains respected even on a CUDA host.
        rife_module.device = self.device
        self.model = rife_module.Model(arbitrary=True)
        self.model.load_model(str(model_dir), -1)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        print(
            f"Loaded RIFE_m from {model_dir}; device={self.device}, "
            f"fp16={self.fp16}, scale={self.scale}"
        )

    def interpolate(self, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
        if timestep <= 1e-6:
            return first
        if timestep >= 1.0 - 1e-6:
            return second
        if first.shape != second.shape:
            raise RuntimeError(
                f"RIFE input shapes differ: {first.shape} vs {second.shape}"
            )

        height, width = first.shape[:2]
        first_tensor = torch_from_bgr(first, self.torch, self.device)
        second_tensor = torch_from_bgr(second, self.torch, self.device)
        padded_height = ((height - 1) // 32 + 1) * 32
        padded_width = ((width - 1) // 32 + 1) * 32
        padding = (0, padded_width - width, 0, padded_height - height)
        first_tensor = self.F.pad(first_tensor, padding)
        second_tensor = self.F.pad(second_tensor, padding)

        with self.torch.inference_mode():
            if self.fp16:
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    middle = self.model.inference(
                        first_tensor,
                        second_tensor,
                        scale=self.scale,
                        timestep=float(timestep),
                    )
            else:
                middle = self.model.inference(
                    first_tensor,
                    second_tensor,
                    scale=self.scale,
                    timestep=float(timestep),
                )

        middle = middle[0, :, :height, :width].clamp(0, 1)
        return (
            (middle.permute(1, 2, 0).float().cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )


def torch_from_bgr(image: np.ndarray, torch_module, device) -> object:
    tensor = torch_module.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(device, non_blocking=True)


class TemporalFrameCache:
    """Small LRU cache for source and RIFE-interpolated frames."""

    def __init__(self, frame_paths: Sequence[Path], interpolator: RIFEInterpolator, max_items: int) -> None:
        self.frame_paths = frame_paths
        self.interpolator = interpolator
        self.max_items = max(4, int(max_items))
        self.frames = OrderedDict()
        self.interpolated = OrderedDict()

    def _put(self, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_items:
            cache.popitem(last=False)

    def frame(self, index: int) -> np.ndarray:
        index = int(index)
        if index in self.frames:
            value = self.frames.pop(index)
            self.frames[index] = value
            return value
        value = read_image(self.frame_paths[index])
        self._put(self.frames, index, value)
        return value

    def at(self, timestamp: float) -> np.ndarray:
        timestamp = float(np.clip(timestamp, 0, len(self.frame_paths) - 1))
        left = int(np.floor(timestamp))
        alpha = timestamp - left
        if alpha <= 1e-6 or left >= len(self.frame_paths) - 1:
            return self.frame(left)

        key = (left, round(alpha, 6))
        if key in self.interpolated:
            value = self.interpolated.pop(key)
            self.interpolated[key] = value
            return value

        first = self.frame(left)
        second = self.frame(left + 1)
        value = self.interpolator.interpolate(first, second, alpha)
        self._put(self.interpolated, key, value)
        return value


def integrate_subframes(
    subframes: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    result = np.zeros_like(subframes[0], dtype=np.float32)
    for subframe, weight in zip(subframes, weights):
        result += np.float32(weight) * subframe.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def integrate_center_frame(
    center: np.ndarray,
    flow_to_previous: Optional[np.ndarray],
    flow_to_next: Optional[np.ndarray],
    offsets: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    result = np.zeros_like(center, dtype=np.float32)
    height, width = center.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    for offset, weight in zip(offsets, weights):
        offset = float(offset)
        if abs(offset) < 1e-8:
            subframe = center
        elif offset < 0 and flow_to_previous is not None:
            subframe = warp_by_fraction(
                center, flow_to_previous, -offset, grid_x, grid_y
            )
        elif offset > 0 and flow_to_next is not None:
            subframe = warp_by_fraction(center, flow_to_next, offset, grid_x, grid_y)
        else:
            # Sequence boundary: edge replication.  This preserves exposure
            # sample count and brightness when an adjacent frame is absent.
            subframe = center
        result += np.float32(weight) * subframe.astype(np.float32)

    result = np.clip(result, 0.0, 255.0).astype(np.uint8)
    return result


def save_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(
            str(path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    else:
        ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed to write blurred image: {path}")


def should_write(path: Path, resume: bool, overwrite: bool) -> bool:
    if not path.exists():
        return True
    if resume:
        return False
    if overwrite:
        return True
    raise FileExistsError(
        f"output already exists: {path}; use --resume or --overwrite"
    )


def generate_sequence(
    sequence_dir: Path,
    output_sequence_dir: Path,
    offsets: Sequence[float],
    weights: Sequence[float],
    flow_params: dict,
    jpeg_quality: int,
    resume: bool,
    overwrite: bool,
    frame_limit: Optional[int],
    progress_desc: str,
) -> dict:
    frame_paths = numeric_image_files(sequence_dir)
    if frame_limit is not None:
        frame_paths = frame_paths[:frame_limit]
    if not frame_paths:
        raise RuntimeError(f"no frames found in {sequence_dir}")

    generated = 0
    skipped = 0
    progress = tqdm(
        total=len(frame_paths),
        desc=progress_desc,
        unit="frame",
        leave=False,
    )

    first = read_image(frame_paths[0])
    if len(frame_paths) == 1:
        output_path = output_sequence_dir / frame_paths[0].name
        if should_write(output_path, resume, overwrite):
            save_image(output_path, first, jpeg_quality)
            generated += 1
        else:
            skipped += 1
        progress.update(1)
        progress.close()
        return {"frames": 1, "generated": generated, "skipped": skipped}

    next_image = read_image(frame_paths[1])
    check_same_shape(first, next_image, frame_paths[0], frame_paths[1])
    flow_current_to_next, flow_next_to_current = estimate_farneback_pair(
        first,
        next_image,
        **flow_params,
    )

    # Frame 0: there is no previous frame, so negative exposure samples use
    # edge replication and positive samples use flow to frame 1.
    output_path = output_sequence_dir / frame_paths[0].name
    if should_write(output_path, resume, overwrite):
        blurred = integrate_center_frame(
            first,
            flow_to_previous=None,
            flow_to_next=flow_current_to_next,
            offsets=offsets,
            weights=weights,
        )
        save_image(output_path, blurred, jpeg_quality)
        generated += 1
    else:
        skipped += 1
    progress.update(1)

    # This is the flow from frame 1 to frame 0.  It becomes the previous-flow
    # input when frame 1 is generated.
    flow_to_previous_for_next_image = flow_next_to_current

    # At the beginning of each iteration, current is frame i and
    # flow_to_previous_for_next_image is the flow i -> i-1.
    for index in range(1, len(frame_paths) - 1):
        next_next_image = read_image(frame_paths[index + 1])
        check_same_shape(next_image, next_next_image, frame_paths[index], frame_paths[index + 1])
        flow_current_to_next, flow_next_to_current = estimate_farneback_pair(
            next_image,
            next_next_image,
            **flow_params,
        )

        output_path = output_sequence_dir / frame_paths[index].name
        if should_write(output_path, resume, overwrite):
            blurred = integrate_center_frame(
                next_image,
                flow_to_previous=flow_to_previous_for_next_image,
                flow_to_next=flow_current_to_next,
                offsets=offsets,
                weights=weights,
            )
            save_image(output_path, blurred, jpeg_quality)
            generated += 1
        else:
            skipped += 1
        progress.update(1)

        next_image = next_next_image
        # The reverse flow of the pair just computed is the previous-flow for
        # the next center frame.
        flow_to_previous_for_next_image = flow_next_to_current

    # After the loop, next_image is the final frame and the carried state is
    # the flow from that final frame to its previous frame.  For a two-frame
    # sequence it is the reverse flow computed before the loop.
    final = next_image
    output_path = output_sequence_dir / frame_paths[-1].name
    if should_write(output_path, resume, overwrite):
        blurred = integrate_center_frame(
            final,
            flow_to_previous=flow_to_previous_for_next_image,
            flow_to_next=None,
            offsets=offsets,
            weights=weights,
        )
        save_image(output_path, blurred, jpeg_quality)
        generated += 1
    else:
        skipped += 1
    progress.update(1)
    progress.close()

    return {"frames": len(frame_paths), "generated": generated, "skipped": skipped}


def generate_sequence_rife(
    sequence_dir: Path,
    output_sequence_dir: Path,
    offsets: Sequence[float],
    weights: Sequence[float],
    interpolator: RIFEInterpolator,
    jpeg_quality: int,
    resume: bool,
    overwrite: bool,
    frame_limit: Optional[int],
    cache_size: int,
    progress_desc: str,
) -> dict:
    """Generate a sequence using actual adjacent-frame RIFE interpolation."""

    frame_paths = numeric_image_files(sequence_dir)
    if frame_limit is not None:
        frame_paths = frame_paths[:frame_limit]
    if not frame_paths:
        raise RuntimeError(f"no frames found in {sequence_dir}")

    cache = TemporalFrameCache(frame_paths, interpolator, cache_size)
    generated = 0
    skipped = 0
    progress = tqdm(
        total=len(frame_paths),
        desc=progress_desc,
        unit="frame",
        leave=False,
    )

    for center_index, frame_path in enumerate(frame_paths):
        output_path = output_sequence_dir / frame_path.name
        if should_write(output_path, resume, overwrite):
            subframes = [
                cache.at(center_index + float(offset)) for offset in offsets
            ]
            blurred = integrate_subframes(subframes, weights)
            save_image(output_path, blurred, jpeg_quality)
            generated += 1
        else:
            skipped += 1
        progress.update(1)

    progress.close()
    return {
        "frames": len(frame_paths),
        "generated": generated,
        "skipped": skipped,
    }


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
        help="Prepared DREB root containing the blur_images directories.",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--flow-backend",
        choices=("farneback", "rife"),
        default="rife",
        help="Interpolation backend; RIFE uses actual adjacent-frame arbitrary-timestep interpolation.",
    )
    parser.add_argument(
        "--rife-root",
        type=Path,
        default=Path("third_party/ECCV2022-RIFE"),
        help="Local official ECCV2022-RIFE repository root.",
    )
    parser.add_argument(
        "--rife-model-dir",
        type=Path,
        default=None,
        help="Directory containing the RIFE_m flownet.pkl checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="RIFE device. Use cuda to fail fast if GPU support is unavailable.",
    )
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--rife-fp16", action="store_true", help="Use CUDA autocast FP16 for RIFE inference.")
    parser.add_argument("--interp-cache-size", type=int, default=32)
    parser.add_argument("--exposure-span", type=float, default=1.0, help="Exposure duration in source-frame units.")
    parser.add_argument("--num-subframes", type=int, default=7, help="Odd number of virtual subframes.")
    parser.add_argument("--weight-mode", choices=("uniform", "triangular"), default="uniform")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--pyr-scale", type=float, default=0.5)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--winsize", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--poly-n", type=int, default=5)
    parser.add_argument("--poly-sigma", type=float, default=1.2)
    parser.add_argument("--resume", action="store_true", help="Skip already generated images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing blurred images.")
    parser.add_argument("--limit-sequences", type=int, default=None, help="Process only the first N sequences.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Process only the first N frames per sequence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")
    if args.limit_sequences is not None and args.limit_sequences < 1:
        raise ValueError("limit-sequences must be positive")
    if args.limit_frames is not None and args.limit_frames < 1:
        raise ValueError("limit-frames must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    offsets, weights = make_offsets_and_weights(
        args.exposure_span,
        args.num_subframes,
        args.weight_mode,
    )
    flow_params = {
        "pyr_scale": args.pyr_scale,
        "levels": args.levels,
        "winsize": args.winsize,
        "iterations": args.iterations,
        "poly_n": args.poly_n,
        "poly_sigma": args.poly_sigma,
    }

    rife_interpolator = None
    if args.flow_backend == "rife":
        if args.rife_model_dir is None:
            raise ValueError("--rife-model-dir is required when --flow-backend rife is selected")
        rife_interpolator = RIFEInterpolator(
            rife_root=args.rife_root.expanduser().resolve(),
            model_dir=args.rife_model_dir.expanduser().resolve(),
            device_name=args.device,
            scale=args.rife_scale,
            fp16=args.rife_fp16,
        )
    if args.flow_backend == "farneback" and args.exposure_span > 2:
        print(
            "WARNING: exposure-span > 2 uses extrapolation of the nearest "
            "adjacent optical flow; validate visual quality before training."
        )

    for split in args.splits:
        source_sequences_root = dataset_root / f"VisDrone2019-VID-{split}" / "sequences"
        if not source_sequences_root.is_dir():
            raise FileNotFoundError(f"missing source sequence directory: {source_sequences_root}")

        sequences = sorted(path for path in source_sequences_root.iterdir() if path.is_dir())
        if args.limit_sequences is not None:
            sequences = sequences[:args.limit_sequences]
        if not sequences:
            raise RuntimeError(f"no sequences found in {source_sequences_root}")

        split_output_root = output_root / split
        split_output_root.mkdir(parents=True, exist_ok=True)
        config = {
            "dataset_root": str(dataset_root),
            "split": split,
            "flow_backend": args.flow_backend,
            "flow_direction": "source_to_neighbor",
            "interpolation_method": (
                "rife_pairwise_arbitrary_timestep"
                if args.flow_backend == "rife"
                else "fractional_center_warp"
            ),
            "exposure_span_in_frames": args.exposure_span,
            "number_of_subframes": args.num_subframes,
            "temporal_offsets": [float(value) for value in offsets],
            "temporal_weights": [float(value) for value in weights],
            "boundary_policy": "edge_replication",
            "jpeg_quality": args.jpeg_quality,
            "farneback": flow_params,
            "rife_root": str(args.rife_root.expanduser().resolve()) if args.flow_backend == "rife" else None,
            "rife_model_dir": str(args.rife_model_dir.expanduser().resolve()) if args.rife_model_dir else None,
            "device": args.device,
            "rife_scale": args.rife_scale,
            "rife_fp16": args.rife_fp16,
            "interp_cache_size": args.interp_cache_size,
            "limit_sequences": args.limit_sequences,
            "limit_frames": args.limit_frames,
        }
        (split_output_root / "blur_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        split_summary = {
            "split": split,
            "sequences": len(sequences),
            "frames": 0,
            "generated": 0,
            "skipped": 0,
        }
        with tqdm(sequences, desc=f"{split} sequences", unit="sequence") as sequence_progress:
            for sequence_dir in sequence_progress:
                sequence_progress.set_postfix(sequence=sequence_dir.name)
                output_sequence_dir = split_output_root / "blur_images" / sequence_dir.name
                if args.flow_backend == "rife":
                    result = generate_sequence_rife(
                        sequence_dir=sequence_dir,
                        output_sequence_dir=output_sequence_dir,
                        offsets=offsets,
                        weights=weights,
                        interpolator=rife_interpolator,
                        jpeg_quality=args.jpeg_quality,
                        resume=args.resume,
                        overwrite=args.overwrite,
                        frame_limit=args.limit_frames,
                        cache_size=args.interp_cache_size,
                        progress_desc=f"{split}/{sequence_dir.name}",
                    )
                else:
                    result = generate_sequence(
                        sequence_dir=sequence_dir,
                        output_sequence_dir=output_sequence_dir,
                        offsets=offsets,
                        weights=weights,
                        flow_params=flow_params,
                        jpeg_quality=args.jpeg_quality,
                        resume=args.resume,
                        overwrite=args.overwrite,
                        frame_limit=args.limit_frames,
                        progress_desc=f"{split}/{sequence_dir.name}",
                    )
                for key in ("frames", "generated", "skipped"):
                    split_summary[key] += result[key]
                tqdm.write(json.dumps({"sequence": sequence_dir.name, **result}))

        (split_output_root / "blur_summary.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(split_summary))


if __name__ == "__main__":
    main()
