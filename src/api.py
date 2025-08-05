from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
from io import BytesIO

from model import SegmentationModel
from predict import load_model, predict_in_patches, clean_mask_via_morph_ops

app = FastAPI()

@app.post("/run-inference")
async def run_inference(image: UploadFile = File(...), type: str = Query(...)):
    contents = await image.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if type == "wall":
        model = load_model(model_ckpt="model.ckpt", gpu_id=0)
        predicted_mask = predict_in_patches(img, model)
        cleaned_mask = clean_mask_via_morph_ops(predicted_mask)
        # Return the cleaned mask as a response
        _, buffer = cv2.imencode('.png', cleaned_mask)
        headers={"filename": image.filename, "status": "ok"}
        return Response(content=buffer.tobytes(), media_type="image/png", headers=headers)
    
    else:
        result = {"error": f"Unsupported type: {type}"}

    return JSONResponse(content=result)