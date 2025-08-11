import cv2
import numpy as np
import onnxruntime


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
    if image.shape[2] == 3: # Change to CxHxW format
        image = image.transpose(2, 0, 1) 
    image = image / 255.0  # Normalize to [0, 1]

    # Pad to next multiple of 32
    h, w = image.shape[1:3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Preprocess the image
    input_np_tensor = image[np.newaxis, :].astype(np.float32)
    input_name = onnx_session.get_inputs()[0].name # The model is expected to have only one input
    pred = onnx_session.run([], {input_name: input_np_tensor})
    pred = pred.squeeze(0) # Remove the batch dim

    if pred.shape[0]==1: # Support binary and multi-class segmentation
        pred_mask = (pred > 0).astype('uint8') * 255
    else:
        pred_mask = np.argmax(pred, axis=1)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[:h, :w]
    return pred_mask

def load_model(onnx_model_path="model.onnx"):
    """Load the trained segmentation model from checkpoint.
    Args:
        model_ckpt: Path to the model checkpoint
    Returns:
        Loaded model
    """
    session = onnxruntime.InferenceSession(onnx_model_path,
                                           providers=["CUDAExecutionProvider",
                                                      "CPUExecutionProvider"])

    return session