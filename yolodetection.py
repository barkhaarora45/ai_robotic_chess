import time
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

# ---------------- USER CONFIG ----------------
# Path to your trained best weights (change to your path)
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\best.pt"   # <- change this to your best.pt path

# Webcam index (0 is default internal webcam). Or set to video file path.
WEBCAM_INDEX = 0

# Confidence threshold (use the value from your F1 curve; you said ~0.37)
CONF_THRESHOLD = 0.387

# Image size to send to YOLO (640 is fine)
IMG_SIZE = 864

# Whether to save the output video
SAVE_OUTPUT = False
OUT_VIDEO_PATH = "live_output.mp4"

# ------------------------------------------------

# sanity checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load model (this will also move it to GPU if available and requested)
model = YOLO(MODEL_PATH)

# If GPU is available, ultralytics uses it by default. Print device info:
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU (this will be slower)")

# Open webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam/index {WEBCAM_INDEX}")

# Optional: prepare writer if saving
writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (w, h))
    print("Saving output to:", OUT_VIDEO_PATH)

# Helper to draw boxes on frame (using result.boxes)
def draw_boxes(frame, boxes, names):
    """
    boxes: ultralytics Boxes object in result.boxes
    names: model.names dict
    """
    if boxes is None or len(boxes) == 0:
        return frame
    # xyxy, cls, conf
    xyxy = boxes.xyxy.cpu().numpy()    # shape (N,4)
    confs = boxes.conf.cpu().numpy()   # shape (N,)
    clss = boxes.cls.cpu().numpy().astype(int)  # shape (N,)
    for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, clss):
        # convert to ints
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{names.get(c, str(c))} {conf:.2f}"
        # choose color by class id (stable)
        color = (int((c * 37) % 255), int((c * 17) % 255), int((c * 97) % 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # text bg
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame

print("Starting live detection. Press 'q' or ESC to quit.")
prev_time = 0.0
try:
    # Stream mode: yields results per frame (stream=True)
    # We pass conf (confidence), imgsz and device will be handled by ultralytics
    stream = model.predict(source=WEBCAM_INDEX, stream=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)

    for result in stream:
        # result.orig_img is the original BGR frame (numpy)
        frame = result.orig_img  # already numpy BGR
        if frame is None:
            continue

        # Draw boxes manually (so we control font/color)
        frame = draw_boxes(frame, result.boxes, model.model.names)  # model.model.names or model.names works

        # FPS calc
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show
        cv2.imshow("YOLO Live", frame)

        # Write to file if required
        if writer is not None:
            writer.write(frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Exiting.")
