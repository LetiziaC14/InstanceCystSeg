from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from nibabel.processing import resample_from_to, resample_to_output
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes


def two_largest_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask)
    if count < 2:
        raise ValueError("Kidney mask must contain at least two components")
    sizes = np.bincount(labels.ravel())
    largest = np.argsort(sizes[1:])[-2:] + 1
    components = [labels == label for label in largest]
    components.sort(key=lambda component: np.argwhere(component)[:, 0].mean())
    return components[0], components[1]


def world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    voxel_centroid = np.argwhere(mask).mean(axis=0)
    return nib.affines.apply_affine(affine, voxel_centroid)


def mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    smoothing_sigma: float,
    target_faces: int,
    surface_smoothing_iterations: int = 0,
    signed_distance: bool = False,
) -> trimesh.Trimesh:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot create a mesh from an empty mask")
    lower = np.maximum(coordinates.min(axis=0) - 2, 0)
    upper = np.minimum(coordinates.max(axis=0) + 3, mask.shape)
    slices = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    cropped = mask[slices]

    local_affine = affine.copy()
    local_affine[:3, 3] = nib.affines.apply_affine(affine, lower)
    padded_mask = np.pad(cropped.astype(bool), 2)
    if signed_distance:
        inside = ndimage.distance_transform_edt(padded_mask)
        outside = ndimage.distance_transform_edt(~padded_mask)
        surface_field = inside - outside
        if smoothing_sigma > 0:
            surface_field = ndimage.gaussian_filter(
                surface_field,
                smoothing_sigma,
            )
        target_voxels = int(np.count_nonzero(padded_mask))
        flat = surface_field.ravel()
        threshold_index = max(0, flat.size - target_voxels)
        level = float(np.partition(flat, threshold_index)[threshold_index])
    else:
        surface_field = padded_mask.astype(np.float32)
        if smoothing_sigma > 0:
            smoothed = ndimage.gaussian_filter(
                surface_field,
                smoothing_sigma,
            )
            if np.max(smoothed) >= 0.5:
                surface_field = smoothed
        level = 0.5

    vertices, faces, _, _ = marching_cubes(surface_field, level=level)
    vertices -= 2.0
    vertices = nib.affines.apply_affine(local_affine, vertices)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    if target_faces > 0 and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    if surface_smoothing_iterations > 0 and len(mesh.vertices) >= 20:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.45,
            nu=0.47,
            iterations=surface_smoothing_iterations,
        )
        mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def material(
    name: str,
    texture: Image.Image | None,
    alpha_mode: str,
    roughness: float,
    base_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=texture,
        baseColorFactor=base_color,
        metallicFactor=0.0,
        roughnessFactor=roughness,
        alphaMode=alpha_mode,
        doubleSided=True,
    )


def organic_texture(
    base_color: tuple[int, int, int],
    alpha: int,
    seed: int,
    contrast: float,
    size: int = 256,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    coarse = ndimage.gaussian_filter(
        rng.normal(size=(size, size)).astype(np.float32),
        sigma=size / 18.0,
        mode="wrap",
    )
    fine = ndimage.gaussian_filter(
        rng.normal(size=(size, size)).astype(np.float32),
        sigma=size / 80.0,
        mode="wrap",
    )
    pattern = 0.75 * coarse / max(float(np.std(coarse)), 1e-6)
    pattern += 0.25 * fine / max(float(np.std(fine)), 1e-6)
    pattern = np.clip(pattern * contrast, -0.32, 0.32)

    base = np.asarray(base_color, dtype=np.float32)
    rgb = base[None, None, :] * (1.0 + pattern[:, :, None])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha_channel = np.full((size, size, 1), alpha, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, alpha_channel], axis=2), "RGBA")


def spherical_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1)
    radius = np.maximum(radius, 1e-8)
    unit = centered / radius[:, None]
    u = np.arctan2(unit[:, 1], unit[:, 0]) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(unit[:, 2], -1.0, 1.0)) / np.pi
    return np.column_stack([u, v])


def apply_texture(
    mesh: trimesh.Trimesh,
    texture_material: trimesh.visual.material.PBRMaterial,
) -> None:
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=spherical_uv(mesh),
        material=texture_material,
    )


def assign_cysts(
    instances: np.ndarray,
    right_kidney: np.ndarray,
    left_kidney: np.ndarray,
    affine: np.ndarray,
    association_dilation: int,
    minimum_volume_mm3: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    right_region = ndimage.binary_dilation(
        right_kidney,
        iterations=association_dilation,
    )
    left_region = ndimage.binary_dilation(
        left_kidney,
        iterations=association_dilation,
    )
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    measurements: list[dict[str, object]] = []

    labels, counts = np.unique(instances[instances > 0], return_counts=True)
    for label, count in zip(labels, counts):
        volume_mm3 = float(count * voxel_volume)
        if volume_mm3 < minimum_volume_mm3:
            continue
        cyst = instances == label
        right_overlap = int(np.count_nonzero(cyst & right_region))
        left_overlap = int(np.count_nonzero(cyst & left_region))
        if max(right_overlap, left_overlap) == 0:
            continue

        side = "right" if right_overlap >= left_overlap else "left"
        kidney_region = right_region if side == "right" else left_region
        retained = cyst & kidney_region
        centroid = world_centroid(retained, affine)
        measurements.append(
            {
                "label": int(label),
                "kidney": side,
                "voxel_count": int(np.count_nonzero(retained)),
                "volume_mm3": float(np.count_nonzero(retained) * voxel_volume),
                "centroid_x_mm": float(centroid[0]),
                "centroid_y_mm": float(centroid[1]),
                "centroid_z_mm": float(centroid[2]),
            }
        )
    return right_region, left_region, measurements


def isotropic_label_grid(
    kidney_image: nib.Nifti1Image,
    cyst_image: nib.Nifti1Image,
    spacing_mm: float,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    if spacing_mm <= 0:
        raise ValueError("--isotropic-spacing must be positive")

    kidney_isotropic = resample_to_output(
        kidney_image,
        voxel_sizes=(spacing_mm, spacing_mm, spacing_mm),
        order=0,
    )
    cyst_isotropic = resample_from_to(
        cyst_image,
        kidney_isotropic,
        order=0,
    )
    return kidney_isotropic, cyst_isotropic


def cyst_color(index: int) -> tuple[int, int, int]:
    palette = (
        (220, 72, 78),
        (238, 102, 62),
        (214, 132, 72),
        (186, 72, 112),
        (226, 94, 126),
        (172, 58, 72),
        (232, 82, 48),
        (164, 78, 132),
    )
    return palette[index % len(palette)]


def create_cyst_meshes(
    instances: np.ndarray,
    right_region: np.ndarray,
    left_region: np.ndarray,
    measurements: list[dict[str, object]],
    affine: np.ndarray,
    smoothing_sigma: float,
    total_target_faces: int,
    surface_smoothing_iterations: int,
    minimum_isotropic_volume_mm3: float,
    isotropic_spacing_mm: float,
) -> tuple[list[tuple[str, trimesh.Trimesh]], set[int]]:
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    exported_labels: set[int] = set()
    for index, row in enumerate(measurements):
        label = int(row["label"])
        kidney_region = (
            right_region if row["kidney"] == "right" else left_region
        )
        mask = instances == label
        if not np.any(mask & kidney_region):
            continue

        components, component_count = ndimage.label(mask)
        if component_count > 1:
            sizes = np.bincount(components.ravel())
            largest = int(np.argmax(sizes[1:]) + 1)
            mask = components == largest
        mask = ndimage.binary_fill_holes(mask)

        isotropic_volume = (
            np.count_nonzero(mask) * isotropic_spacing_mm**3
        )
        if isotropic_volume < minimum_isotropic_volume_mm3:
            continue

        mesh = mesh_from_mask(
            mask,
            affine,
            smoothing_sigma,
            0,
            surface_smoothing_iterations=surface_smoothing_iterations,
            signed_distance=True,
        )
        name = f"Cyst_{label:04d}_{row['kidney']}"
        meshes.append((name, mesh))
        exported_labels.add(label)

    raw_faces = sum(len(mesh.faces) for _, mesh in meshes)
    if total_target_faces > 0 and raw_faces > total_target_faces:
        simplified: list[tuple[str, trimesh.Trimesh]] = []
        for name, mesh in meshes:
            target = max(
                20,
                int(round(total_target_faces * len(mesh.faces) / raw_faces)),
            )
            if len(mesh.faces) > target:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            mesh.fix_normals()
            simplified.append((name, mesh))
        meshes = simplified

    cyst_materials = [
        material(
            f"Cyst tissue {index + 1}",
            organic_texture(
                cyst_color(index),
                alpha=242,
                seed=300 + index,
                contrast=0.10,
            ),
            "BLEND",
            roughness=0.18,
        )
        for index in range(8)
    ]
    for index, (_, mesh) in enumerate(meshes):
        apply_texture(mesh, cyst_materials[index % len(cyst_materials)])
    return meshes, exported_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export kidney and cyst segmentations as a colored GLB model"
    )
    parser.add_argument("--kidney-mask", type=Path, required=True)
    parser.add_argument("--cyst-instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--kidney-opacity", type=float, default=1.0)
    parser.add_argument("--kidney-faces", type=int, default=60000)
    parser.add_argument(
        "--cyst-faces",
        type=int,
        default=0,
        help="Total cyst face budget; 0 preserves full closed surfaces",
    )
    parser.add_argument("--isotropic-spacing", type=float, default=1.0)
    parser.add_argument("--minimum-cyst-volume-mm3", type=float, default=50.0)
    parser.add_argument("--kidney-smoothing-sigma-mm", type=float, default=1.25)
    parser.add_argument("--cyst-smoothing-sigma-mm", type=float, default=1.2)
    parser.add_argument("--kidney-smoothing-iterations", type=int, default=30)
    parser.add_argument("--cyst-smoothing-iterations", type=int, default=15)
    parser.add_argument("--association-dilation", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.kidney_opacity <= 1.0:
        raise ValueError("--kidney-opacity must be in (0, 1]")

    kidney_image = nib.load(args.kidney_mask)
    cyst_image = nib.load(args.cyst_instances)
    if kidney_image.shape != cyst_image.shape or not np.allclose(
        kidney_image.affine,
        cyst_image.affine,
        rtol=1e-4,
        atol=1e-3,
    ):
        kidney_voxel_volume = abs(
            np.linalg.det(kidney_image.affine[:3, :3])
        )
        cyst_voxel_volume = abs(
            np.linalg.det(cyst_image.affine[:3, :3])
        )
        if kidney_voxel_volume <= cyst_voxel_volume:
            cyst_image = resample_from_to(
                cyst_image,
                kidney_image,
                order=0,
            )
            print("Resampled cyst labels onto the finer kidney-mask grid")
        else:
            kidney_image = resample_from_to(
                kidney_image,
                cyst_image,
                order=0,
            )
            print("Resampled kidney mask onto the finer cyst-instance grid")

    source_kidney = np.asanyarray(kidney_image.dataobj) > 0
    source_instances = np.asanyarray(cyst_image.dataobj).astype(np.int32)
    source_first, source_second = two_largest_components(source_kidney)

    # NIfTI world coordinates use RAS: larger x is anatomical right.
    if world_centroid(source_first, kidney_image.affine)[0] > world_centroid(
        source_second,
        kidney_image.affine,
    )[0]:
        source_right_kidney, source_left_kidney = source_first, source_second
    else:
        source_right_kidney, source_left_kidney = source_second, source_first

    _, _, measurements = assign_cysts(
        source_instances,
        source_right_kidney,
        source_left_kidney,
        kidney_image.affine,
        args.association_dilation,
        args.minimum_cyst_volume_mm3,
    )
    if not measurements:
        raise ValueError("No cyst instances overlap the reconstructed kidneys")

    retained_labels = np.asarray(
        [int(row["label"]) for row in measurements],
        dtype=np.int32,
    )
    filtered_instances = np.where(
        np.isin(source_instances, retained_labels),
        source_instances,
        0,
    ).astype(np.int32)
    filtered_cyst_image = nib.Nifti1Image(
        filtered_instances,
        kidney_image.affine,
    )
    isotropic_kidney_image, isotropic_cyst_image = isotropic_label_grid(
        nib.Nifti1Image(
            source_kidney.astype(np.uint8),
            kidney_image.affine,
        ),
        filtered_cyst_image,
        args.isotropic_spacing,
    )

    kidney = np.asanyarray(isotropic_kidney_image.dataobj) > 0
    instances = np.rint(
        np.asanyarray(isotropic_cyst_image.dataobj)
    ).astype(np.int32)
    first, second = two_largest_components(kidney)
    if world_centroid(first, isotropic_kidney_image.affine)[0] > world_centroid(
        second,
        isotropic_kidney_image.affine,
    )[0]:
        right_kidney, left_kidney = first, second
    else:
        right_kidney, left_kidney = second, first

    right_region = ndimage.binary_dilation(
        right_kidney,
        iterations=max(1, int(round(2.0 / args.isotropic_spacing))),
    )
    left_region = ndimage.binary_dilation(
        left_kidney,
        iterations=max(1, int(round(2.0 / args.isotropic_spacing))),
    )
    measurement_by_label = {
        int(row["label"]): row for row in measurements
    }
    present_labels = set(np.unique(instances[instances > 0]).tolist())
    measurements = [
        measurement_by_label[label]
        for label in sorted(measurement_by_label)
        if label in present_labels
    ]

    kidney_sigma_voxels = (
        args.kidney_smoothing_sigma_mm / args.isotropic_spacing
    )
    cyst_sigma_voxels = (
        args.cyst_smoothing_sigma_mm / args.isotropic_spacing
    )
    right_mesh = mesh_from_mask(
        right_kidney,
        isotropic_kidney_image.affine,
        kidney_sigma_voxels,
        args.kidney_faces,
        surface_smoothing_iterations=args.kidney_smoothing_iterations,
    )
    left_mesh = mesh_from_mask(
        left_kidney,
        isotropic_kidney_image.affine,
        kidney_sigma_voxels,
        args.kidney_faces,
        surface_smoothing_iterations=args.kidney_smoothing_iterations,
    )
    cyst_meshes, exported_labels = create_cyst_meshes(
        instances,
        right_region,
        left_region,
        measurements,
        isotropic_kidney_image.affine,
        cyst_sigma_voxels,
        args.cyst_faces,
        args.cyst_smoothing_iterations,
        args.minimum_cyst_volume_mm3,
        args.isotropic_spacing,
    )
    measurements = [
        row for row in measurements if int(row["label"]) in exported_labels
    ]
    if not cyst_meshes:
        raise ValueError("No cyst surfaces remain after regularization")

    alpha = int(round(args.kidney_opacity * 255))
    kidney_alpha_mode = "OPAQUE" if alpha == 255 else "BLEND"
    right_material = material(
        "Kidney tissue",
        None,
        kidney_alpha_mode,
        roughness=0.62,
        base_color=(150, 58, 54, alpha),
    )
    left_material = material(
        "Kidney tissue",
        None,
        kidney_alpha_mode,
        roughness=0.62,
        base_color=(150, 58, 54, alpha),
    )
    apply_texture(right_mesh, right_material)
    apply_texture(left_mesh, left_material)
    scene = trimesh.Scene()
    scene.add_geometry(right_mesh, node_name="Right kidney")
    scene.add_geometry(left_mesh, node_name="Left kidney")
    for name, cyst_mesh in cyst_meshes:
        scene.add_geometry(cyst_mesh, node_name=name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output, file_type="glb")

    measurements_path = args.measurements or args.output.with_suffix(".csv")
    with measurements_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)

    total_cyst_volume = sum(row["volume_mm3"] for row in measurements)
    print(f"Saved {args.output}")
    print(f"Saved {measurements_path}")
    print(
        f"Meshes: right kidney={len(right_mesh.faces)} faces, "
        f"left kidney={len(left_mesh.faces)} faces, "
        f"cysts={sum(len(mesh.faces) for _, mesh in cyst_meshes)} faces "
        f"across {len(cyst_meshes)} separate objects"
    )
    print(
        f"Included {len(measurements)} cyst instances, "
        f"total cyst volume={total_cyst_volume:.1f} mm3"
    )
    print(
        f"Reconstruction grid: {args.isotropic_spacing:.2f} mm isotropic; "
        f"minimum cyst volume: {args.minimum_cyst_volume_mm3:.1f} mm3"
    )


if __name__ == "__main__":
    main()
