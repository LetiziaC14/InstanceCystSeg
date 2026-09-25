from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage
from skimage.morphology import ball


def largest_components(mask: np.ndarray, count: int = 2) -> list[np.ndarray]:
    labels, total = ndimage.label(mask)
    if total < count:
        raise ValueError(f"Expected at least {count} kidney components")
    sizes = np.bincount(labels.ravel())
    selected = np.argsort(sizes[1:])[-count:] + 1
    return [labels == label for label in selected]


def volume_preserving_smooth(
    mask: np.ndarray,
    sigma_voxels: float,
) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot smooth an empty kidney component")

    margin = max(4, int(np.ceil(4.0 * sigma_voxels)))
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, mask.shape)
    slices = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    cropped = mask[slices]

    filled = ndimage.binary_fill_holes(cropped)
    inside = ndimage.distance_transform_edt(filled)
    outside = ndimage.distance_transform_edt(~filled)
    signed_distance = inside - outside
    if sigma_voxels > 0:
        signed_distance = ndimage.gaussian_filter(
            signed_distance,
            sigma_voxels,
        )

    target_voxels = int(np.count_nonzero(filled))
    flat = signed_distance.ravel()
    threshold_index = max(0, flat.size - target_voxels)
    threshold = float(np.partition(flat, threshold_index)[threshold_index])
    smoothed = signed_distance >= threshold
    smoothed = ndimage.binary_fill_holes(smoothed)

    labels, total = ndimage.label(smoothed)
    if total > 1:
        sizes = np.bincount(labels.ravel())
        smoothed = labels == (np.argmax(sizes[1:]) + 1)

    result = np.zeros(mask.shape, dtype=bool)
    result[slices] = smoothed
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smooth isotropic two-kidney mask"
    )
    parser.add_argument("--kidney-mask", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="NIfTI defining the target isotropic grid",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoothing-sigma-mm",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--isotropic-spacing",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--opening-radius-mm",
        type=float,
        default=0.0,
        help="Remove narrow attached protrusions before smoothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kidney_image = nib.load(args.kidney_mask)
    reference = nib.load(args.reference)
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    if not np.allclose(spacing, spacing[0], rtol=1e-4, atol=1e-4):
        reference = resample_to_output(
            reference,
            voxel_sizes=(args.isotropic_spacing,) * 3,
            order=1,
        )
        spacing = np.asarray(
            reference.header.get_zooms()[:3],
            dtype=np.float64,
        )
        print(
            "Created isotropic reference grid:",
            reference.shape,
            tuple(spacing),
        )
    if args.smoothing_sigma_mm < 0:
        raise ValueError("--smoothing-sigma-mm cannot be negative")
    if args.opening_radius_mm < 0:
        raise ValueError("--opening-radius-mm cannot be negative")

    resampled = resample_from_to(kidney_image, reference, order=0)
    kidney = np.asanyarray(resampled.dataobj) > 0
    components = largest_components(kidney, 2)
    if args.opening_radius_mm > 0:
        radius_voxels = max(
            1,
            int(round(args.opening_radius_mm / float(spacing[0]))),
        )
        opened_components = []
        for component in components:
            opened = ndimage.binary_opening(
                component,
                structure=ball(radius_voxels),
            )
            opened_components.append(largest_components(opened, 1)[0])
        components = opened_components
        print(
            f"Applied spherical opening: {args.opening_radius_mm:.2f} mm "
            f"({radius_voxels} voxels)"
        )

    sigma_voxels = args.smoothing_sigma_mm / float(spacing[0])
    smoothed_components = [
        volume_preserving_smooth(component, sigma_voxels)
        for component in components
    ]
    output_mask = np.logical_or.reduce(smoothed_components)

    voxel_volume = abs(np.linalg.det(reference.affine[:3, :3]))
    input_volume = sum(np.count_nonzero(component) for component in components)
    output_volume = np.count_nonzero(output_mask)

    output = nib.Nifti1Image(
        output_mask.astype(np.uint8),
        reference.affine,
        reference.header,
    )
    output.set_data_dtype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, args.output)

    print(f"Saved {args.output}")
    print(
        f"Grid: {output.shape}, spacing={tuple(spacing)} mm, "
        f"sigma={args.smoothing_sigma_mm:.2f} mm"
    )
    print(
        f"Kidney volume: input={input_volume * voxel_volume:.1f} mm3, "
        f"output={output_volume * voxel_volume:.1f} mm3"
    )


if __name__ == "__main__":
    main()
