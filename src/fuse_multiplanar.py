from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


BACKGROUND = 0
CORE = 1
EDGE = 2


def read_float(path: Path) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat32)


def read_mask(path: Path) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)


def read_labels(path: Path) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkUInt8)


def isotropic_reference(
    image: sitk.Image,
    spacing_mm: float,
) -> sitk.Image:
    if spacing_mm <= 0:
        raise ValueError("--isotropic-spacing must be positive")
    source_size = np.asarray(image.GetSize(), dtype=np.float64)
    source_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    physical_extent = (source_size - 1.0) * source_spacing
    target_size = np.ceil(physical_extent / spacing_mm).astype(int) + 1

    reference = sitk.Image(
        [int(value) for value in target_size],
        sitk.sitkFloat32,
    )
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    return reference


def identity_resample(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    pixel_type: int | None = None,
) -> sitk.Image:
    return resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        pixel_type=pixel_type,
    )


def largest_components(mask: sitk.Image, count: int) -> sitk.Image:
    components = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = sorted(
        stats.GetLabels(),
        key=stats.GetNumberOfPixels,
        reverse=True,
    )[:count]
    result = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    result.CopyInformation(mask)
    for label in labels:
        result = result | sitk.Cast(components == label, sitk.sitkUInt8)
    return result


def component_by_lps_side(mask: sitk.Image, side: str) -> sitk.Image:
    components = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = sorted(
        stats.GetLabels(),
        key=stats.GetNumberOfPixels,
        reverse=True,
    )[:2]
    if len(labels) != 2:
        raise ValueError("The fixed kidney mask must contain two large components")

    # DICOM/SimpleITK uses LPS coordinates: positive x is anatomical left.
    key = max if side == "left" else min
    selected = key(labels, key=lambda label: stats.GetCentroid(label)[0])
    return sitk.Cast(components == selected, sitk.sitkUInt8)


def mask_centroid(mask: sitk.Image) -> tuple[float, float, float]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(mask))
    labels = stats.GetLabels()
    if not labels:
        raise ValueError("Registration mask is empty")

    weighted = np.zeros(3, dtype=np.float64)
    total = 0
    for label in labels:
        size = stats.GetNumberOfPixels(label)
        weighted += np.asarray(stats.GetCentroid(label)) * size
        total += size
    return tuple(weighted / total)


def robust_normalize(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_array = sitk.GetArrayViewFromImage(mask) > 0
    values = array[mask_array]
    if values.size == 0:
        raise ValueError("Cannot normalize an image with an empty mask")
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        raise ValueError("MRI intensity range is degenerate")
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(image)
    return result


def initial_rigid_transform(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
) -> sitk.Euler3DTransform:
    fixed_center = np.asarray(mask_centroid(fixed_mask))
    moving_center = np.asarray(mask_centroid(moving_mask))
    transform = sitk.Euler3DTransform()
    transform.SetCenter(tuple(fixed_center))
    transform.SetTranslation(tuple(moving_center - fixed_center))
    return transform


def mask_dice(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:
    registered = resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    fixed_array = sitk.GetArrayViewFromImage(fixed_mask) > 0
    registered_array = sitk.GetArrayViewFromImage(registered) > 0
    denominator = np.count_nonzero(fixed_array) + np.count_nonzero(
        registered_array
    )
    if denominator == 0:
        return 0.0
    return float(
        2.0
        * np.count_nonzero(fixed_array & registered_array)
        / denominator
    )


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    name: str,
) -> sitk.Transform:
    fixed_normalized = robust_normalize(fixed, fixed_mask)
    moving_normalized = robust_normalize(moving, moving_mask)
    initial = initial_rigid_transform(fixed_mask, moving_mask)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.20, 42)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=250,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=20,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel((4, 2, 1))
    registration.SetSmoothingSigmasPerLevel((2, 1, 0))
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)

    optimized = registration.Execute(fixed_normalized, moving_normalized)
    initial_dice = mask_dice(fixed_mask, moving_mask, initial)
    optimized_dice = mask_dice(fixed_mask, moving_mask, optimized)
    if optimized_dice < initial_dice:
        transform = initial
        selection = "initial centroid alignment"
    else:
        transform = optimized
        selection = "optimized mutual-information alignment"
    print(
        f"{name}: metric={registration.GetMetricValue():.6f}, "
        f"iterations={registration.GetOptimizerIteration()}, "
        f"initial_dice={initial_dice:.4f}, "
        f"optimized_dice={optimized_dice:.4f}, "
        f"selected={selection}, "
        f"stop={registration.GetOptimizerStopConditionDescription()}"
    )
    return transform


def resample(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
    interpolator: int,
    default_value: float = 0.0,
    pixel_type: int | None = None,
) -> sitk.Image:
    if pixel_type is None:
        pixel_type = moving.GetPixelID()
    return sitk.Resample(
        moving,
        fixed,
        transform,
        interpolator,
        default_value,
        pixel_type,
    )


def combine_sagittal(
    right: sitk.Image,
    left: sitk.Image,
    right_valid: sitk.Image,
    left_valid: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    right_array = sitk.GetArrayFromImage(right)
    left_array = sitk.GetArrayFromImage(left)
    right_mask = sitk.GetArrayFromImage(right_valid) > 0
    left_mask = sitk.GetArrayFromImage(left_valid) > 0

    combined = np.zeros(right_array.shape, dtype=np.uint8)
    combined[right_mask] = right_array[right_mask]
    only_left = left_mask & ~right_mask
    combined[only_left] = left_array[only_left]

    overlap = right_mask & left_mask
    agreement = overlap & (right_array == left_array)
    combined[agreement] = right_array[agreement]
    disagreement = overlap & ~agreement
    combined[disagreement] = BACKGROUND

    valid = right_mask | left_mask
    combined_image = sitk.GetImageFromArray(combined)
    combined_image.CopyInformation(right)
    valid_image = sitk.GetImageFromArray(valid.astype(np.uint8))
    valid_image.CopyInformation(right)
    return combined_image, valid_image


def full_coverage(image: sitk.Image) -> sitk.Image:
    result = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    result.CopyInformation(image)
    return result + 1


def combine_images(
    right: sitk.Image,
    left: sitk.Image,
    right_coverage: sitk.Image,
    left_coverage: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    right_array = sitk.GetArrayFromImage(right).astype(np.float32)
    left_array = sitk.GetArrayFromImage(left).astype(np.float32)
    right_valid = sitk.GetArrayFromImage(right_coverage) > 0
    left_valid = sitk.GetArrayFromImage(left_coverage) > 0
    denominator = right_valid.astype(np.float32) + left_valid.astype(np.float32)
    combined = np.divide(
        right_array * right_valid + left_array * left_valid,
        denominator,
        out=np.zeros_like(right_array),
        where=denominator > 0,
    )
    coverage = denominator > 0

    combined_image = sitk.GetImageFromArray(combined)
    combined_image.CopyInformation(right)
    coverage_image = sitk.GetImageFromArray(coverage.astype(np.uint8))
    coverage_image.CopyInformation(right)
    return combined_image, coverage_image


def majority_vote(
    predictions: list[sitk.Image],
    valid_masks: list[sitk.Image],
) -> tuple[sitk.Image, sitk.Image, sitk.Image]:
    prediction_arrays = [
        sitk.GetArrayFromImage(image).astype(np.uint8) for image in predictions
    ]
    valid_arrays = [
        sitk.GetArrayFromImage(mask) > 0 for mask in valid_masks
    ]
    voters = np.sum(valid_arrays, axis=0).astype(np.uint8)
    counts = np.stack(
        [
            np.sum(
                [
                    valid & (prediction == label)
                    for prediction, valid in zip(
                        prediction_arrays,
                        valid_arrays,
                    )
                ],
                axis=0,
            )
            for label in (BACKGROUND, CORE, EDGE)
        ],
        axis=0,
    )
    winner = np.argmax(counts, axis=0).astype(np.uint8)
    winning_votes = np.max(counts, axis=0)

    required = np.maximum(2, voters // 2 + 1)
    consensus = (voters >= 2) & (winning_votes >= required)
    fused = np.where(consensus, winner, BACKGROUND).astype(np.uint8)

    reference = predictions[0]
    outputs = []
    for array in (fused, voters, consensus.astype(np.uint8)):
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(reference)
        outputs.append(image)
    return tuple(outputs)


def create_instances(
    image: sitk.Image,
    semantic: sitk.Image,
) -> sitk.Image:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage

    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    semantic_array = sitk.GetArrayFromImage(semantic)
    core = semantic_array == CORE
    if not np.any(core):
        result = sitk.Image(semantic.GetSize(), sitk.sitkInt32)
        result.CopyInformation(semantic)
        return result

    shape_distance = ndimage.distance_transform_edt(
        core,
        sampling=tuple(reversed(semantic.GetSpacing())),
    )
    image_scale = max(float(np.max(image_array)), 1e-6)
    distance_scale = max(float(np.max(shape_distance)), 1e-6)
    weighted = (image_array / image_scale) * (
        shape_distance / distance_scale
    )

    spacing_zyx = tuple(reversed(semantic.GetSpacing()))
    footprint = tuple(
        max(1, int(round(20.0 / spacing))) for spacing in spacing_zyx
    )
    coordinates = peak_local_max(
        weighted,
        footprint=np.ones(footprint, dtype=bool),
        labels=core,
        exclude_border=False,
    )
    marker_mask = np.zeros(core.shape, dtype=bool)
    marker_mask[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(marker_mask)
    labels = watershed(-weighted, markers, mask=core).astype(np.int32)

    missing = core & (labels == 0)
    connected, _ = ndimage.label(missing)
    if connected.max():
        connected[connected > 0] += labels.max()
    labels += connected.astype(np.int32)

    final = np.zeros_like(labels, dtype=np.int32)
    counts = np.bincount(labels.ravel())
    for label in np.argsort(counts[1:]) + 1:
        if counts[label] <= 1:
            continue
        final[ndimage.binary_dilation(labels == label)] = label

    result = sitk.GetImageFromArray(final)
    result.CopyInformation(semantic)
    return sitk.Cast(result, sitk.sitkInt32)


def fuse_mri(
    images: list[sitk.Image],
    valid_masks: list[sitk.Image],
) -> sitk.Image:
    arrays = [
        sitk.GetArrayFromImage(image).astype(np.float32) for image in images
    ]
    masks = [sitk.GetArrayFromImage(mask) > 0 for mask in valid_masks]
    numerator = np.sum(
        [array * mask for array, mask in zip(arrays, masks)],
        axis=0,
    )
    denominator = np.sum(masks, axis=0)
    fused = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    result = sitk.GetImageFromArray(fused.astype(np.float32))
    result.CopyInformation(images[0])
    return result


def write(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and fuse coronal, axial, and sagittal predictions"
    )
    parser.add_argument("--coronal-image", type=Path, required=True)
    parser.add_argument("--coronal-kidney", type=Path, required=True)
    parser.add_argument("--coronal-semantic", type=Path, required=True)
    parser.add_argument("--axial-image", type=Path, required=True)
    parser.add_argument("--axial-kidney", type=Path, required=True)
    parser.add_argument("--axial-semantic", type=Path, required=True)
    parser.add_argument("--sag-right-image", type=Path, required=True)
    parser.add_argument("--sag-right-kidney", type=Path, required=True)
    parser.add_argument("--sag-right-semantic", type=Path, required=True)
    parser.add_argument("--sag-left-image", type=Path, required=True)
    parser.add_argument("--sag-left-kidney", type=Path, required=True)
    parser.add_argument("--sag-left-semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--isotropic-spacing",
        type=float,
        help="Optional output-grid spacing in mm; registration and voting use this grid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fixed_source = read_float(args.coronal_image)
    fixed_kidney_source = read_mask(args.coronal_kidney)
    coronal_semantic_source = read_labels(args.coronal_semantic)

    if args.isotropic_spacing:
        fixed = isotropic_reference(fixed_source, args.isotropic_spacing)
        fixed = identity_resample(
            fixed_source,
            fixed,
            sitk.sitkLinear,
            sitk.sitkFloat32,
        )
        fixed_kidney_raw = identity_resample(
            fixed_kidney_source,
            fixed,
            sitk.sitkNearestNeighbor,
            sitk.sitkUInt8,
        )
        coronal_semantic = identity_resample(
            coronal_semantic_source,
            fixed,
            sitk.sitkNearestNeighbor,
            sitk.sitkUInt8,
        )
        print(
            "Fusion grid:",
            fixed.GetSize(),
            "spacing:",
            fixed.GetSpacing(),
        )
    else:
        fixed = fixed_source
        fixed_kidney_raw = fixed_kidney_source
        coronal_semantic = coronal_semantic_source

    fixed_kidney = largest_components(fixed_kidney_raw, 2)
    fixed_right = component_by_lps_side(fixed_kidney, "right")
    fixed_left = component_by_lps_side(fixed_kidney, "left")

    axial = read_float(args.axial_image)
    axial_kidney = largest_components(read_mask(args.axial_kidney), 2)
    axial_transform = register_rigid(
        fixed,
        axial,
        fixed_kidney,
        axial_kidney,
        "axial_to_coronal",
    )
    sitk.WriteTransform(
        axial_transform,
        str(output / "axial_to_coronal.tfm"),
    )

    right = read_float(args.sag_right_image)
    right_kidney = largest_components(read_mask(args.sag_right_kidney), 1)
    right_transform = register_rigid(
        fixed,
        right,
        fixed_right,
        right_kidney,
        "sag_right_to_coronal",
    )
    sitk.WriteTransform(
        right_transform,
        str(output / "sag_right_to_coronal.tfm"),
    )

    left = read_float(args.sag_left_image)
    left_kidney = largest_components(read_mask(args.sag_left_kidney), 1)
    left_transform = register_rigid(
        fixed,
        left,
        fixed_left,
        left_kidney,
        "sag_left_to_coronal",
    )
    sitk.WriteTransform(
        left_transform,
        str(output / "sag_left_to_coronal.tfm"),
    )

    axial_valid = resample(
        axial_kidney,
        fixed,
        axial_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    axial_registered = robust_normalize(
        resample(axial, fixed, axial_transform, sitk.sitkLinear),
        axial_valid,
    )
    axial_coverage = resample(
        full_coverage(axial),
        fixed,
        axial_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    axial_semantic = resample(
        read_labels(args.axial_semantic),
        fixed,
        axial_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )

    right_valid = resample(
        right_kidney,
        fixed,
        right_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    right_registered = robust_normalize(
        resample(right, fixed, right_transform, sitk.sitkLinear),
        right_valid,
    )
    right_coverage = resample(
        full_coverage(right),
        fixed,
        right_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    right_semantic = resample(
        read_labels(args.sag_right_semantic),
        fixed,
        right_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )

    left_valid = resample(
        left_kidney,
        fixed,
        left_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    left_registered = robust_normalize(
        resample(left, fixed, left_transform, sitk.sitkLinear),
        left_valid,
    )
    left_coverage = resample(
        full_coverage(left),
        fixed,
        left_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )
    left_semantic = resample(
        read_labels(args.sag_left_semantic),
        fixed,
        left_transform,
        sitk.sitkNearestNeighbor,
        pixel_type=sitk.sitkUInt8,
    )

    sagittal_semantic, sagittal_valid = combine_sagittal(
        right_semantic,
        left_semantic,
        right_valid,
        left_valid,
    )
    sagittal_image, sagittal_coverage = combine_images(
        right_registered,
        left_registered,
        right_coverage,
        left_coverage,
    )

    fixed_normalized = robust_normalize(fixed, fixed_kidney)
    fused_semantic, voter_count, consensus = majority_vote(
        [coronal_semantic, axial_semantic, sagittal_semantic],
        [fixed_kidney, axial_valid, sagittal_valid],
    )
    fused_image = fuse_mri(
        [fixed_normalized, axial_registered, sagittal_image],
        [full_coverage(fixed), axial_coverage, sagittal_coverage],
    )
    fused_instances = create_instances(fused_image, fused_semantic)

    write(axial_registered, output / "axial_registered.nii.gz")
    write(axial_semantic, output / "axial_semantic_registered.nii.gz")
    write(right_registered, output / "sag_right_registered.nii.gz")
    write(right_semantic, output / "sag_right_semantic_registered.nii.gz")
    write(left_registered, output / "sag_left_registered.nii.gz")
    write(left_semantic, output / "sag_left_semantic_registered.nii.gz")
    write(sagittal_semantic, output / "sagittal_semantic_combined.nii.gz")
    write(voter_count, output / "voter_count.nii.gz")
    write(consensus, output / "consensus_mask.nii.gz")
    write(fused_image, output / "multiplanar_fused_mri.nii.gz")
    write(fused_semantic, output / "multiplanar_semantic_majority.nii.gz")
    write(fused_instances, output / "multiplanar_cyst_instances.nii.gz")

    instance_array = sitk.GetArrayViewFromImage(fused_instances)
    semantic_array = sitk.GetArrayViewFromImage(fused_semantic)
    voters_array = sitk.GetArrayViewFromImage(voter_count)
    labels = np.unique(instance_array)
    print(f"Detected {np.count_nonzero(labels)} fused cyst instances")
    print(
        "Semantic counts:",
        dict(zip(*np.unique(semantic_array, return_counts=True))),
    )
    print(
        "Voter counts:",
        dict(zip(*np.unique(voters_array, return_counts=True))),
    )


if __name__ == "__main__":
    main()
