# Copyright (c) 2025 András Kalapos
# Licensed under the MIT License. See LICENSE file in the project root for details.

import os
import cv2
import numpy as np
import onnxruntime


def preference_bias(logits, preferred_classes, strength=0.1):
    """Bias scaled to local logit range"""
    logit_std = logits.std()
    bias_value = strength * logit_std

    bias = np.zeros_like(logits)
    for c in preferred_classes:
        bias[c, ...] = bias_value

    return logits + bias


def predict(image, onnx_session):
    """Run prediction on a single image using the trained model.
    Args:
        image: Input image as a numpy array
        onnx_session: ONNX Runtime session for inference
    Returns:
        Predicted mask as a numpy array
    """
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:  # Change to CxHxW format
        image = image.transpose(2, 0, 1)
    image = image / 255.0  # Normalize to [0, 1]

    # Pad to 512
    h, w = image.shape[1:3]
    pad_h = 512 - h
    pad_w = 512 - w
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    # Preprocess the image
    input_np_tensor = image[np.newaxis, :].astype(np.float32)
    input_name = onnx_session.get_inputs()[
        0
    ].name  # The model is expected to have only one input
    pred = onnx_session.run([], {input_name: input_np_tensor})
    pred = pred[
        0
    ]  # Get the first output, the model is expected to have only one output
    pred = pred.squeeze(0)  # Remove the batch dim

    if pred.shape[0] == 1:  # Support binary and multi-class segmentation
        pred_mask = (pred > 0).squeeze().astype("uint8") * 255
    else:
        pred = preference_bias(pred, preferred_classes=[1, 2], strength=1.2)
        pred_mask = np.argmax(pred, axis=0)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[:h, :w]
    return pred_mask


def check_gpu_availability():
    """
    Check if CUDAExecutionProvider is available and
    if the GPU is actually accessible.
    """
    gpu_available = False
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        try:
            # Try to detect GPU devices
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
            )
            gpu_available = result.returncode == 0 and "GPU" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            gpu_available = False
    return gpu_available


def load_model(onnx_model_path="model.onnx"):
    """Load the trained segmentation model from checkpoint.
    Args:
        model_ckpt: Path to the model checkpoint
    Returns:
        Loaded model
    """
    assert os.path.exists(onnx_model_path), (
        f"Model file {onnx_model_path} does not exist."
    )

    if check_gpu_availability():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    return session
