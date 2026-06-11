from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import nibabel as nib
import numpy as np
from keras import Model, layers
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


MODEL_SIZE = 512
Z_FACTOR = 3


class LayerNames:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        value = self.counts.get(prefix, 0) + 1
        self.counts[prefix] = value
        return f"{prefix}_{value}"


def build_model() -> Model:
    names = LayerNames()

    def conv(
        filters: int,
        kernel_size: tuple[int, int],
        *,
        strides: tuple[int, int] = (1, 1),
        padding: str = "same",
        activation: str | None = None,
        initializer: str = "glorot_uniform",
    ) -> layers.Conv2D:
        return layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=initializer,
            name=names.next("conv2d"),
        )

    def elu(x):
        return layers.ELU(alpha=1.0, name=names.next("elu"))(x)

    def batch_norm(x):
        return layers.BatchNormalization(axis=-1, name=names.next("batch_normalization"))(x)

    def inception_block(x, depth: int):
        c1 = conv(depth // 4, (1, 1), initializer="he_normal")(x)

        c2 = conv(depth // 8 * 3, (1, 1), initializer="he_normal")(x)
        c2 = elu(c2)
        c2 = conv(depth // 2, (3, 3), initializer="he_normal")(c2)

        c3 = conv(depth // 16, (1, 1), initializer="he_normal")(x)
        c3 = elu(c3)
        c3 = conv(depth // 8, (5, 5), initializer="he_normal")(c3)

        pooled = layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=names.next("max_pooling2d"),
        )(x)
        c4 = conv(depth // 8, (1, 1), initializer="he_normal")(pooled)

        result = layers.Concatenate(axis=-1, name=names.next("concatenate"))(
            [c1, c2, c3, c4]
        )
        return elu(batch_norm(result))

    def downsample(x, filters: int):
        x = conv(filters, (3, 3), strides=(2, 2))(x)
        return elu(batch_norm(x))

    def residual_block(x, depth: int):
        residual = conv(depth, (1, 1))(x)
        residual = batch_norm(residual)
        residual = layers.Lambda(
            lambda value: value * 0.1,
            name=names.next("lambda"),
        )(residual)
        x = layers.Add(name=names.next("add"))([x, residual])
        return elu(x)

    def dropout(x):
        return layers.Dropout(0.1, name=names.next("dropout"))(x)

    inputs = layers.Input((MODEL_SIZE, MODEL_SIZE, 4), name="input_1")

    conv1 = inception_block(inputs, 64)
    pool1 = dropout(downsample(conv1, 64))

    conv2 = inception_block(pool1, 128)
    pool2 = dropout(downsample(conv2, 128))

    conv3 = inception_block(pool2, 256)
    pool3 = dropout(downsample(conv3, 256))

    conv4 = inception_block(pool3, 512)
    pool4 = dropout(downsample(conv4, 512))

    conv5 = dropout(inception_block(pool4, 1024))

    up6 = layers.Concatenate(axis=-1, name=names.next("concatenate"))(
        [
            layers.UpSampling2D(
                size=(2, 2),
                name=names.next("up_sampling2d"),
            )(conv5),
            residual_block(conv4, 512),
        ]
    )
    conv6 = dropout(inception_block(up6, 512))

    up7 = layers.Concatenate(axis=-1, name=names.next("concatenate"))(
        [
            layers.UpSampling2D(
                size=(2, 2),
                name=names.next("up_sampling2d"),
            )(conv6),
            residual_block(conv3, 256),
        ]
    )
    conv7 = dropout(inception_block(up7, 256))

    up8 = layers.Concatenate(axis=-1, name=names.next("concatenate"))(
        [
            layers.UpSampling2D(
                size=(2, 2),
                name=names.next("up_sampling2d"),
            )(conv7),
            residual_block(conv2, 128),
        ]
    )
    conv8 = dropout(inception_block(up8, 128))

    up9 = layers.Concatenate(axis=-1, name=names.next("concatenate"))(
        [
            layers.UpSampling2D(
                size=(2, 2),
                name=names.next("up_sampling2d"),
            )(conv8),
            residual_block(conv1, 64),
        ]
    )
    conv9 = dropout(inception_block(up9, 64))

    outputs = conv(3, (1, 1), padding="valid", activation="sigmoid")(conv9)
    return Model(inputs=inputs, outputs=outputs, name="instance_cyst_seg")


def load_legacy_weights(model: Model, checkpoint: Path) -> None:
    loaded = 0
    with h5py.File(checkpoint, "r") as handle:
        root = handle["model_weights"]
        for layer in model.layers:
            if not layer.weights:
                continue
            if layer.name not in root:
                raise ValueError(f"Checkpoint does not contain layer {layer.name!r}")

            datasets: list[np.ndarray] = []
            layer_group = root[layer.name]
            weight_names = layer_group.attrs.get("weight_names", [])
            group = (
                layer_group[layer.name]
                if layer.name in layer_group
                else layer_group
            )
            for weight_name in weight_names:
                if isinstance(weight_name, bytes):
                    weight_name = weight_name.decode("utf-8")
                short_name = weight_name.split("/")[-1]
                datasets.append(np.asarray(group[short_name]))

            expected = [tuple(weight.shape) for weight in layer.weights]
            actual = [tuple(value.shape) for value in datasets]
            if expected != actual:
                raise ValueError(
                    f"Weight mismatch for {layer.name}: expected {expected}, got {actual}"
                )
            layer.set_weights(datasets)
            loaded += 1

    print(f"Loaded {loaded} weighted layers from {checkpoint}")


def validate_inputs(image: nib.Nifti1Image, kidney: nib.Nifti1Image) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"MRI must be 3-D, got shape {image.shape}")
    if image.shape != kidney.shape:
        raise ValueError(
            f"MRI and kidney mask shapes differ: {image.shape} vs {kidney.shape}"
        )
    if not np.allclose(image.affine, kidney.affine, rtol=1e-4, atol=1e-2):
        raise ValueError("MRI and kidney mask affines are not compatible")


def prepare_volume(
    image_data: np.ndarray,
    kidney_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.percentile(image_data, 99)
    if scale <= 0:
        raise ValueError("MRI 99th percentile must be positive")
    image_data = np.clip(image_data / scale * 255.0, 0.0, 255.0)
    kidney_data = (kidney_data > 0).astype(np.float32)

    image_rotated = np.rot90(image_data, axes=(0, 2))
    kidney_rotated = np.rot90(kidney_data, axes=(0, 2))
    in_plane_size = kidney_data.shape[0]
    interpolated_slices = kidney_data.shape[2] * Z_FACTOR

    image_resampled = np.zeros(
        (interpolated_slices, in_plane_size, image_rotated.shape[2]),
        dtype=np.float32,
    )
    kidney_resampled = np.zeros_like(image_resampled)
    for index in range(image_rotated.shape[2]):
        image_resampled[:, :, index] = cv2.resize(
            image_rotated[:, :, index],
            (in_plane_size, interpolated_slices),
            interpolation=cv2.INTER_CUBIC,
        )
        kidney_resampled[:, :, index] = cv2.resize(
            kidney_rotated[:, :, index],
            (in_plane_size, interpolated_slices),
            interpolation=cv2.INTER_NEAREST,
        )

    image_stack = np.rot90(image_resampled, 3, axes=(0, 2))
    kidney_stack = np.rot90(kidney_resampled, 3, axes=(0, 2))
    return image_stack, kidney_stack


def resized_image_stack(image_stack: np.ndarray) -> np.ndarray:
    result = np.empty(
        (image_stack.shape[2], MODEL_SIZE, MODEL_SIZE),
        dtype=np.float32,
    )
    for index in range(image_stack.shape[2]):
        result[index] = cv2.resize(
            image_stack[:, :, index],
            (MODEL_SIZE, MODEL_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
    return result


def prepare_reformatted_volume(
    image: nib.Nifti1Image,
    kidney: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    source_spacing = nib.affines.voxel_sizes(image.affine)
    target_spacing = float(min(source_spacing))

    image_isotropic = resample_to_output(
        image,
        voxel_sizes=(target_spacing,) * 3,
        order=3,
    )
    kidney_isotropic = resample_from_to(
        kidney,
        image_isotropic,
        order=0,
    )

    start_orientation = nib.orientations.io_orientation(image_isotropic.affine)
    target_orientation = nib.orientations.axcodes2ornt(("L", "I", "P"))
    transform = nib.orientations.ornt_transform(
        start_orientation,
        target_orientation,
    )

    image_data = nib.orientations.apply_orientation(
        image_isotropic.get_fdata(dtype=np.float32),
        transform,
    )
    kidney_data = nib.orientations.apply_orientation(
        kidney_isotropic.get_fdata(dtype=np.float32),
        transform,
    )
    target_affine = image_isotropic.affine @ nib.orientations.inv_ornt_aff(
        transform,
        image_isotropic.shape,
    )
    target_image = nib.Nifti1Image(image_data, target_affine)

    scale = np.percentile(image_data, 99)
    if scale <= 0:
        raise ValueError("MRI 99th percentile must be positive")
    image_data = np.clip(image_data / scale * 255.0, 0.0, 255.0)
    kidney_data = kidney_data > 0

    return image_data, kidney_data.astype(np.float32), target_image


def reformatted_stacks(
    image_data: np.ndarray,
    kidney_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image_stack = np.empty(
        (image_data.shape[2], MODEL_SIZE, MODEL_SIZE),
        dtype=np.float32,
    )
    kidney_stack = np.empty_like(image_stack)
    for index in range(image_data.shape[2]):
        image_stack[index] = cv2.resize(
            image_data[:, :, index],
            (MODEL_SIZE, MODEL_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
        kidney_stack[index] = cv2.resize(
            kidney_data[:, :, index],
            (MODEL_SIZE, MODEL_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
    return image_stack, kidney_stack


def predict_semantic(
    model: Model,
    image_stack: np.ndarray,
    kidney_stack: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = resized_image_stack(image_stack)
    mean = float(np.mean(images))
    std = float(np.std(images))
    if std == 0:
        raise ValueError("MRI standard deviation is zero after preprocessing")
    normalized = (images - mean) / std
    normalized_zero = np.float32(-mean / std)

    semantic = np.empty(images.shape, dtype=np.uint8)
    total = images.shape[0]
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = np.empty((stop - start, MODEL_SIZE, MODEL_SIZE, 4), np.float32)
        for offset, index in enumerate(range(start, stop)):
            batch[offset, :, :, 0] = (
                normalized[index - 1] if index else normalized_zero
            )
            batch[offset, :, :, 1] = normalized[index]
            batch[offset, :, :, 2] = (
                normalized[index + 1]
                if index + 1 < total
                else normalized_zero
            )
            batch[offset, :, :, 3] = cv2.resize(
                kidney_stack[:, :, index],
                (MODEL_SIZE, MODEL_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
        probabilities = model(batch, training=False).numpy()
        semantic[start:stop] = np.argmax(probabilities, axis=-1).astype(np.uint8)
        print(f"Predicted slices {stop}/{total}")

    return images, semantic


def predict_semantic_from_resized(
    model: Model,
    images: np.ndarray,
    kidneys: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    mean = float(np.mean(images))
    std = float(np.std(images))
    if std == 0:
        raise ValueError("MRI standard deviation is zero after preprocessing")
    normalized = (images - mean) / std
    normalized_zero = np.float32(-mean / std)

    semantic = np.empty(images.shape, dtype=np.uint8)
    total = images.shape[0]
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = np.empty((stop - start, MODEL_SIZE, MODEL_SIZE, 4), np.float32)
        for offset, index in enumerate(range(start, stop)):
            batch[offset, :, :, 0] = (
                normalized[index - 1] if index else normalized_zero
            )
            batch[offset, :, :, 1] = normalized[index]
            batch[offset, :, :, 2] = (
                normalized[index + 1]
                if index + 1 < total
                else normalized_zero
            )
            batch[offset, :, :, 3] = kidneys[index]
        probabilities = model(batch, training=False).numpy()
        semantic[start:stop] = np.argmax(probabilities, axis=-1).astype(np.uint8)
        print(f"Predicted slices {stop}/{total}")
    return semantic


def create_instances(
    image_stack: np.ndarray,
    semantic: np.ndarray,
    voxel_spacing: tuple[float, float, float],
) -> np.ndarray:
    cyst_core = semantic == 1
    if not np.any(cyst_core):
        return np.zeros_like(semantic, dtype=np.int32)

    distance_shape = ndimage.distance_transform_edt(cyst_core)
    image_max = float(np.max(image_stack))
    shape_max = float(np.max(distance_shape))
    weighted_distance = (image_stack / max(image_max, 1e-6)) * (
        distance_shape / max(shape_max, 1e-6)
    )

    footprint_size = tuple(
        max(1, int(round(20.0 / spacing))) for spacing in voxel_spacing
    )
    coordinates = peak_local_max(
        weighted_distance,
        footprint=np.ones(footprint_size, dtype=bool),
        labels=cyst_core,
        exclude_border=False,
    )
    marker_mask = np.zeros(cyst_core.shape, dtype=bool)
    marker_mask[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(marker_mask)
    labels_ws = watershed(-weighted_distance, markers, mask=cyst_core)

    unassigned = cyst_core & (labels_ws == 0)
    labels_cc, _ = ndimage.label(unassigned)
    if labels_cc.max():
        labels_cc[labels_cc > 0] += labels_ws.max()
    labels = labels_ws.astype(np.int32) + labels_cc.astype(np.int32)

    final_labels = np.zeros_like(labels, dtype=np.int32)
    counts = np.bincount(labels.ravel())
    for label in np.argsort(counts[1:]) + 1:
        if counts[label] <= 1:
            continue
        dilated = ndimage.binary_dilation(labels == label)
        final_labels[dilated] = label
    return final_labels


def restore_original_shape(
    volume: np.ndarray,
    original_shape: tuple[int, int, int],
) -> np.ndarray:
    in_plane = np.empty(
        (original_shape[0], original_shape[1], volume.shape[0]),
        dtype=volume.dtype,
    )
    for index in range(volume.shape[0]):
        in_plane[:, :, index] = cv2.resize(
            volume[index],
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    rotated = np.rot90(in_plane, axes=(0, 2))
    restored_rotated = np.empty(
        (original_shape[2], original_shape[1], original_shape[0]),
        dtype=volume.dtype,
    )
    for index in range(rotated.shape[2]):
        restored_rotated[:, :, index] = cv2.resize(
            rotated[:, :, index],
            (original_shape[1], original_shape[2]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.rot90(restored_rotated, 3, axes=(0, 2))


def model_stack_to_nifti(
    volume: np.ndarray,
    target: nib.Nifti1Image,
    dtype: np.dtype,
) -> nib.Nifti1Image:
    restored = np.empty(target.shape, dtype=dtype)
    for index in range(target.shape[2]):
        restored[:, :, index] = cv2.resize(
            volume[index],
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    result = nib.Nifti1Image(restored.astype(dtype), target.affine)
    result.set_data_dtype(dtype)
    return result


def save_nifti(
    data: np.ndarray,
    reference: nib.Nifti1Image,
    path: Path,
    dtype: np.dtype,
) -> None:
    output = nib.Nifti1Image(data.astype(dtype), reference.affine, reference.header)
    output.set_data_dtype(dtype)
    nib.save(output, path)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InstanceCystSeg inference")
    parser.add_argument("--image", type=Path, required=True, help="MRI NIfTI path")
    parser.add_argument(
        "--kidney-mask",
        type=Path,
        required=True,
        help="Binary kidney mask NIfTI path",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--acquisition-plane",
        choices=("coronal", "axial", "sagittal"),
        default="coronal",
        help="Acquisition plane of the input MRI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    image = nib.load(args.image)
    kidney = nib.load(args.kidney_mask)
    validate_inputs(image, kidney)

    model = build_model()
    load_legacy_weights(model, args.checkpoint)

    if args.acquisition_plane == "coronal":
        image_data = image.get_fdata(dtype=np.float32)
        kidney_data = kidney.get_fdata(dtype=np.float32)
        image_stack, kidney_stack = prepare_volume(image_data, kidney_data)
        resized_images, semantic = predict_semantic(
            model,
            image_stack,
            kidney_stack,
            args.batch_size,
        )

        zooms = nib.affines.voxel_sizes(image.affine)
        in_plane_spacing = zooms[0] * image.shape[0] / MODEL_SIZE
        instance_spacing = (
            zooms[2] / Z_FACTOR,
            in_plane_spacing,
            in_plane_spacing,
        )
        instances = create_instances(resized_images, semantic, instance_spacing)

        restored_instances = restore_original_shape(instances, image.shape)
        output_instances = restored_instances
        save_nifti(output_instances, image, args.output, np.int32)

        if args.semantic_output:
            restored_semantic = restore_original_shape(semantic, image.shape)
            save_nifti(restored_semantic, image, args.semantic_output, np.uint8)
    else:
        reformatted_image, reformatted_kidney, target = (
            prepare_reformatted_volume(image, kidney)
        )
        resized_images, resized_kidneys = reformatted_stacks(
            reformatted_image,
            reformatted_kidney,
        )
        semantic = predict_semantic_from_resized(
            model,
            resized_images,
            resized_kidneys,
            args.batch_size,
        )

        target_zooms = nib.affines.voxel_sizes(target.affine)
        instance_spacing = (
            target_zooms[2],
            target_zooms[0] * target.shape[0] / MODEL_SIZE,
            target_zooms[1] * target.shape[1] / MODEL_SIZE,
        )
        instances = create_instances(resized_images, semantic, instance_spacing)

        instance_target = model_stack_to_nifti(instances, target, np.int32)
        instance_original = resample_from_to(instance_target, image, order=0)
        output_instances = np.asanyarray(instance_original.dataobj)
        save_nifti(
            output_instances,
            image,
            args.output,
            np.int32,
        )

        if args.semantic_output:
            semantic_target = model_stack_to_nifti(semantic, target, np.uint8)
            semantic_original = resample_from_to(semantic_target, image, order=0)
            save_nifti(
                np.asanyarray(semantic_original.dataobj),
                image,
                args.semantic_output,
                np.uint8,
            )

    labels = np.unique(output_instances)
    print(f"Detected {len(labels[labels > 0])} cyst instances")


if __name__ == "__main__":
    main()
