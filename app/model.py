import os
import urllib.request
import onnxruntime as ort
import numpy as np
from typing import List, Dict, Any, Optional

MODEL_PATH = os.path.join("models", "fcos_pcb_model.onnx")
MODEL_URL = "https://github.com/tolgaguclu17/PCB-Defect-Detection-Fcos-Api/releases/download/v1.0.0/fcos_pcb_model.onnx"

# Automatically download the model file in the production environment if does not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Model file not found locally. Downloading from GitHub Release...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

CLASS_NAMES: Dict[int, str] = {
    1: "spurious_copper",
    2: "missing_hole",
    3: "mouse_bite",
    4: "open_circuit",
    5: "pinhole",
    6: "scratch",
    7: "short",
    8: "spur",
}

class PCBModelWrapper:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None

    def load_model(self):
        """Loads the ONNX model into memory with Render 512MB RAM optimizations."""
        # Custom ONNX optimizations for the 512 MB RAM limit
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False  # Prevents RAM overload
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

    def clear_model(self):
        """Clears the session."""
        self.session = None

    def predict(self, img_np: np.ndarray, score_thresh: float = 0.60) -> Dict[str, np.ndarray]:
    """
    It performs model inference and returns the
    boxes, scores, and labels that exceed the specified threshold.
    """
        if self.session is None:
            raise RuntimeError("The model has not been loaded. Please call load_model() first.")
    
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_np})
    
        boxes = np.squeeze(outputs[0])   # (N, 4)
        scores = np.squeeze(outputs[1])  # (N,)
        labels = np.squeeze(outputs[2])  # (N,)
    
        keep = scores >= score_thresh
        
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep]
        }
