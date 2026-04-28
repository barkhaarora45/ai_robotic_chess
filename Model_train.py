import os
import sys
import platform
import traceback

# --- 1) Show environment info (PyTorch + CUDA) ---
try:
    import torch
    print("=== Environment Check ===")
    print("Python        :", sys.version.split()[0])
    print("OS            :", platform.system(), platform.release())
    print("PyTorch       :", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda (build):", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU count     :", torch.cuda.device_count())
        print("GPU[0] name   :", torch.cuda.get_device_name(0))
    print("=" * 30)
except Exception as e:
    print("PyTorch import failed. Please install torch first.")
    print(e)
    sys.exit(1)

# --- 2) Import YOLO (Ultralytics) ---
try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not installed. Install with: pip install ultralytics")
    print(e)
    sys.exit(1)

# --- 3) Config you may edit ---
# Use forward slashes to avoid Windows backslash/unicode issues
DATA_YAML = r"C:\Users\Admin\Desktop\finalAICHES\data.yaml"   # <-- change if your path is different
MODEL_WEIGHTS = "yolov8n.pt"                      # try 'yolov8s.pt' later for more accuracy
EPOCHS = 60
IMG_SIZE = 672

# Keep batch moderate to avoid OOM on smaller GPUs; increase if you have more VRAM
BATCH = 4

# On Windows, too many workers can cause issues; 0–2 is usually safe
WORKERS = 2

# Use GPU if available
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Mixed precision (FP16) speeds up on NVIDIA GPUs
USE_HALF = bool(torch.cuda.is_available())

# --- 4) Helper: clean path & existence check ---
def ensure_path(p: str) -> str:
    """
    Use forward slashes or raw strings to avoid Windows unicode escape errors.
    """
    p = p.replace("\\", "/")
    if not os.path.exists(p):
        print(f"[WARN] Path not found: {p}")
    return p

DATA_YAML = ensure_path(DATA_YAML)

# --- 5) Train ---
def main():
    print("Loading model:", MODEL_WEIGHTS)
    model = YOLO(MODEL_WEIGHTS)

    print("Starting training...")
    print(f"  data={DATA_YAML}")
    print(f"  device={DEVICE}  epochs={EPOCHS}  imgsz={IMG_SIZE}  batch={BATCH}  workers={WORKERS}  half={USE_HALF}")

    # You can set project/name to organize runs
    train_kwargs = dict(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        half=USE_HALF,     # use FP16 on GPU
        project="runs/chess",
        name="yolov8n_chess",
        verbose=True
    )

    # Try train; if OOM occurs, auto-retry with smaller batch
    try:
        results = model.train(**train_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and isinstance(BATCH, int) and BATCH > 1:
            print("\n[OOM] CUDA out of memory. Reducing batch size and retrying once...")
            torch.cuda.empty_cache()
            train_kwargs["batch"] = max(1, BATCH // 2)
            print("Retry batch:", train_kwargs["batch"])
            results = model.train(**train_kwargs)
        else:
            print("\n[ERROR] Training failed:")
            traceback.print_exc()
            sys.exit(1)

    # --- 6) Validate (metrics on val set) ---
    print("\nRunning validation on the best weights...")
    try:
        metrics = model.val()
        print("Validation metrics:", metrics)
    except Exception as e:
        print("[WARN] Validation step failed:", e)

    # --- 7) Optional: quick predict on one image if you have it ---
    SAMPLE_IMAGE = r"C:\Users\Admin\Downloads\New_Dataset\test\images\1000027445_jpg.rf.dca8c0fbf49766b80a36b27d3aeb1037.jpg"  # put any image here to test
    if os.path.exists(SAMPLE_IMAGE):
        print("\nRunning a quick prediction on:", SAMPLE_IMAGE)
        try:
            preds = model.predict(source=SAMPLE_IMAGE, save=True, conf=0.5)
            print("Prediction saved to:", preds[0].save_dir if preds else "runs/detect/predict")
        except Exception as e:
            print("[WARN] Prediction failed:", e)
    else:
        print("\n(No SAMPLE_IMAGE found; skip quick prediction.)")

    print("\nDone. Check weights at: runs/chess/yolov8n_chess/weights/best.pt")

if __name__ == "__main__":
    main()
