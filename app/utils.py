import io
import numpy as np
from PIL import Image
from typing import Tuple, List

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, int, int]:
    """
    Converts incoming image bytes to a PIL Image, retrieves their original dimensions,
    resizes them to 640x640, and returns a float32 NumPy array in the format [1, 3, 640, 640].
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = pil_image.size

    image_resized = pil_image.resize(target_size)
    img_np = np.array(image_resized).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0)    # CHW -> NCHW

    return img_np, orig_w, orig_h


def rescale_boxes(boxes: np.ndarray, orig_w: int, orig_h: int, target_size: Tuple[int, int] = (640, 640)) -> List[List[float]]:
    """
    Scales the estimated box coordinates based on the 640x640 size to the original image size.
    """
    scale_x = orig_w / float(target_size[0])
    scale_y = orig_h / float(target_size[1])

    rescaled_boxes = []
    for box in boxes:
        xmin = float(box[0] * scale_x)
        ymin = float(box[1] * scale_y)
        xmax = float(box[2] * scale_x)
        ymax = float(box[3] * scale_y)
        rescaled_boxes.append([xmin, ymin, xmax, ymax])

    return rescaled_boxes