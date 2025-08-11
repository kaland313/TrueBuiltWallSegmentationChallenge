import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response

from predict.common import clean_mask_via_morph_ops, predict_in_patches
from predict.onnx import load_model, predict

app = FastAPI()

@app.post("/run-inference")
async def run_inference(image: UploadFile, type: str)-> Response:
    
    contents = await image.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if type == "wall":
        model = load_model(model_ckpt=os.environ.get("ONNX_MODEL_PATH"), gpu_id=0)
        predicted_mask = predict_in_patches(img, model, pred_fn=predict)
        cleaned_mask = clean_mask_via_morph_ops(predicted_mask)
        # Return the cleaned mask as a response
        _, buffer = cv2.imencode('.png', cleaned_mask)
        headers = {'Content-Disposition': f'attachment; filename="{image.filename}"'}
        return Response(buffer.tobytes(), headers=headers, media_type='image/png')   
    else:
        result = {"error": f"Unsupported type: {type}"}

    return JSONResponse(content=result)


# curl -X POST -F "image=@data/train/images/A-102 .00 - 2ND FLOOR PLAN CROP.png" "http://localhost:3000/run-inference?type=wall"

# uvicorn api:app --host 0.0.0.0 --port 3000
