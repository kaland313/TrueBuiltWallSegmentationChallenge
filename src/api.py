# Copyright (c) 2025 András Kalapos
# Licensed under the MIT License. See LICENSE file in the project root for details.

import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response
from typing import Literal

from predict.common import clean_mask_via_morph_ops, predict_in_patches
from predict.onnx import load_model, predict
from img_proc_utils import watershed_segmentation, colorize_regions

app = FastAPI()


@app.post("/run-inference")
async def run_inference(
    image: UploadFile,
    type: Literal["wall", "room"],
    return_raw_segmentation_ids: bool = False,
) -> Response:
    contents = await image.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if type == "wall":
        # Load the modell and run inference
        model = load_model(os.environ.get("ONNX_MODEL_PATH"))
        predicted_wall_mask = predict_in_patches(img, model, pred_fn=predict)
        cleaned_wall_mask = clean_mask_via_morph_ops(predicted_wall_mask)

        # Scale the image for better visualization
        if not return_raw_segmentation_ids:
            cleaned_wall_mask = (
                cleaned_wall_mask * (255 // cleaned_wall_mask.max())
            ).astype(np.uint8)

        # Return the cleaned mask as a response
        _, buffer = cv2.imencode(".png", cleaned_wall_mask)
        filename = f"{Path(image.filename).stem}_wall.png"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(buffer.tobytes(), headers=headers, media_type="image/png")
    elif type == "room":
        # Load the model and run inference
        model = load_model(os.environ.get("ONNX_MODEL_PATH"))
        predicted_wall_mask = predict_in_patches(img, model, pred_fn=predict)
        cleaned_wall_mask = clean_mask_via_morph_ops(predicted_wall_mask)
        room_mask, _ = watershed_segmentation(cleaned_wall_mask)

        # Colorize the mask for visualization
        if not return_raw_segmentation_ids:
            room_mask = colorize_regions(room_mask)

        # Return the cleaned mask as a response
        _, buffer = cv2.imencode(".png", room_mask)
        filename = f"{Path(image.filename).stem}_room.png"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(buffer.tobytes(), headers=headers, media_type="image/png")
    else:
        return None  # This should never happen, as the type is validated by FastAPI
