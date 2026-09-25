from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np


def largest_component(mask: sitk.Image) -> sitk.Image:
    components = sitk.ConnectedComponent(mask > 0)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = stats.GetLabels()
    if not labels:
        raise ValueError("Kidney mask is empty")
    selected = max(labels, key=stats.GetNumberOfPixels)
    return sitk.Cast(components == selected, sitk.sitkUInt8)


def bilateral_components(
    mask: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    components = sitk.ConnectedComponent(mask > 0)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = sorted(
        stats.GetLabels(),
        key=stats.GetNumberOfPixels,
        reverse=True,
    )[:2]
    if len(labels) != 2:
        raise ValueError("Base mask must contain two kidney components")

    # SimpleITK physical coordinates are LPS: lower x is anatomical right.
    labels.sort(key=lambda label: stats.GetCentroid(label)[0])
    right = sitk.Cast(components == labels[0], sitk.sitkUInt8)
    left = sitk.Cast(components == labels[1], sitk.sitkUInt8)
    return right, left


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace one kidney component with a registered side mask"
    )
    parser.add_argument("--base-mask", type=Path, required=True)
    parser.add_argument("--replacement-mask", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--side", choices=("right", "left"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = sitk.ReadImage(str(args.reference))
    base = sitk.ReadImage(str(args.base_mask))
    replacement = sitk.ReadImage(str(args.replacement_mask))
    transform = sitk.ReadTransform(str(args.transform))

    identity = sitk.Transform(3, sitk.sitkIdentity)
    base_on_reference = sitk.Resample(
        base,
        reference,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    replacement_on_reference = sitk.Resample(
        replacement,
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    base_right, base_left = bilateral_components(base_on_reference)
    replacement_side = largest_component(replacement_on_reference)
    if args.side == "right":
        corrected = replacement_side | base_left
        old_side = base_right
    else:
        corrected = base_right | replacement_side
        old_side = base_left

    old_voxels = int(sitk.GetArrayViewFromImage(old_side).sum())
    replacement_voxels = int(
        sitk.GetArrayViewFromImage(replacement_side).sum()
    )
    overlap = int(
        np.count_nonzero(
            (sitk.GetArrayViewFromImage(old_side) > 0)
            & (sitk.GetArrayViewFromImage(replacement_side) > 0)
        )
    )
    dice = 2.0 * overlap / max(old_voxels + replacement_voxels, 1)

    corrected = sitk.Cast(corrected, sitk.sitkUInt8)
    corrected.CopyInformation(reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(corrected, str(args.output), True)
    print(f"Saved {args.output}")
    print(
        f"Replaced {args.side} kidney: old={old_voxels} voxels, "
        f"replacement={replacement_voxels} voxels, Dice={dice:.4f}"
    )


if __name__ == "__main__":
    main()
