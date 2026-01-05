# app/main.py
# FastAPI service for Satellite Image Classification (base64 image -> class)

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import decode_base64_image, load_model_bundle, predict_image


class PredictRequest(BaseModel):
    image: str  # base64-encoded image (optionally with data URI prefix)


app = FastAPI(title="Satellite Image Classification API", version="1.0.0")

# Load model once at startup
MODEL_PATH = "model/model.pth"
bundle = load_model_bundle(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "classes": bundle.classes, "image_size": bundle.image_size}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        img = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {e}")

    pred_class, probs = predict_image(bundle, img)
    return {"predicted_class": pred_class, "probabilities": probs}
