from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

from src.predict import decode_base64_image, load_model_bundle, predict_image

MODEL_PATH = "model/model.pth"

app = FastAPI(title="Satellite Image Classification API", version="1.0.0")
bundle = load_model_bundle(MODEL_PATH)


class PredictRequest(BaseModel):
    image: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": bundle.classes, "image_size": bundle.image_size}


@app.post("/predict-base64")
def predict(req: PredictRequest) -> dict:
    try:
        img = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    pred_class, probs = predict_image(bundle, img)
    return {"predicted_class": pred_class, "probabilities": probs}


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pred_class, probs = predict_image(bundle, img)
    return {"predicted_class": pred_class, "probabilities": probs}
