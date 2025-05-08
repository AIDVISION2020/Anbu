from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_barcode

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (adjust the path if needed)
model = YOLO("yolov8n.pt")

def decode_barcodes_multi(image):
    results = []

    def decode_and_append(img):
        for code in decode_barcode(img):
            x, y, w, h = code.rect
            results.append({
                "type": "barcode",
                "label": code.type,
                "data": code.data.decode("utf-8"),
                "bbox": [x, y, x + w, y + h]
            })

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    decode_and_append(image)     # Original color image
    decode_and_append(gray)      # Grayscale
    decode_and_append(thresh)    # Simple threshold
    decode_and_append(adaptive)  # Adaptive threshold

    return results

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    detections = []

    # 1. Object Detection with YOLO (unchanged)
    yolo_results = model(image)[0]
    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detections.append({
            "type": "object",
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # 2. Barcode/QR Detection (improved version)
    barcode_detections = decode_barcodes_multi(image)
    detections.extend(barcode_detections)

    return JSONResponse(content={"detections": detections})
