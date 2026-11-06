from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from app.model import PCBModelWrapper, CLASS_NAMES
from app.utils import preprocess_image, rescale_boxes

pcb_model = PCBModelWrapper(model_path="models/fcos_pcb_model.onnx")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Models are loading into memory...")
    pcb_model.load_model()
    print("The model has been successfully loaded! Requests are ready to be accepted.")
    yield
    print("The server is shutting down, memory is being cleared...")
    pcb_model.clear_model()

app = FastAPI(title="pcb_fcos_api", lifespan=lifespan)

class Defect(BaseModel):
    class_name: str
    score: float
    bboxes: List[float]

class PredictResponse(BaseModel):
    message: str
    found_defects: List[Defect]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "The server is up"}

@app.post("/predict", response_model=PredictResponse)
async def make_prediction(
    file: UploadFile = File(...),
    score_thresh: float = Query(0.60, ge=0.1, le=1.0, description="Confidence threshold filter")
):
    # Validate file content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a valid image file (e.g., JPG, PNG)."
        )

    filename = file.filename
    image_bytes = await file.read()

    # Check for empty payload
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        # Preprocessing
        img_np, orig_w, orig_h = preprocess_image(image_bytes, target_size=(640, 640))

        # Inference (Pass dynamic score_thresh parameter)
        predictions = pcb_model.predict(img_np, score_thresh=score_thresh)
        
        raw_boxes = predictions["boxes"]
        scores = predictions["scores"]
        labels = predictions["labels"]

        # Rescale Bounding Box Coordinates
        rescaled_boxes = rescale_boxes(raw_boxes, orig_w, orig_h, target_size=(640, 640))

        # Construct response list
        defects_list = []
        for box, label, score in zip(rescaled_boxes, labels, scores):
            label_id = int(label)
            class_name = CLASS_NAMES.get(label_id, f"Class_{label_id}")

            defects_list.append({
                "class_name": class_name,
                "score": float(score),
                "bboxes": box.tolist() if hasattr(box, "tolist") else list(box)
            })

        return {
            "message": f"File '{filename}' processed. Found {len(defects_list)} defect(s).",
            "found_defects": defects_list
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model inference: {str(e)}"
        )

