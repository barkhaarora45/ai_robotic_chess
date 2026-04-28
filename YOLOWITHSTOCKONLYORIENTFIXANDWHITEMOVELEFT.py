import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os
import chess
import chess.engine

# ---------------- SILENCE YOLO TERMINAL SPAM ----------------
LOGGER.setLevel("ERROR")

# ---------------- USER CONFIG ----------------
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\best.pt"
WEBCAM_INDEX = 0
CONF_THRESHOLD = 0.387
IMG_SIZE = 864

STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

WHITE_WAIT_TIME = 20
ENGINE_TIME = 0.2
# ------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(STOCKFISH_PATH):
    raise FileNotFoundError(f"Stockfish not found: {STOCKFISH_PATH}")

model = YOLO(MODEL_PATH)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ---------- YOLO class → chess piece ----------
CLASS_TO_PIECE = {
    "White-Pawn": "P",
    "White-Knight": "N",
    "White-Bishop": "B",
    "White-Rook": "R",
    "White-Queen": "Q",
    "White-King": "K",
    "Black-Pawn": "p",
    "Black-Knight": "n",
    "Black-Bishop": "b",
    "Black-Rook": "r",
    "Black-Queen": "q",
    "Black-King": "k",
}

# ---------- Draw YOLO boxes only ----------
def draw_boxes(frame, boxes, names):
    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{names.get(c, str(c))} {conf:.2f}"
        color = (int((c * 37) % 255), int((c * 17) % 255), int((c * 97) % 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ---------- Convert detections → chess.Board ----------
def detections_to_board(result, frame_shape):
    board_array = [["." for _ in range(8)] for _ in range(8)]

    if result.boxes is None:
        return None

    h, w = frame_shape[:2]
    xyxy = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c in zip(xyxy, clss):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        file = int(cx / w * 8)
        rank = 7 - int(cy / h * 8)

        if not (0 <= file < 8 and 0 <= rank < 8):
            continue

        class_name = model.names[c]
        piece = CLASS_TO_PIECE.get(class_name)

        if piece:
            board_array[rank][file] = piece

    # Convert to FEN
    fen_rows = []
    for row in board_array:
        empty = 0
        fen = ""
        for cell in row:
            if cell == ".":
                empty += 1
            else:
                if empty:
                    fen += str(empty)
                    empty = 0
                fen += cell
        if empty:
            fen += str(empty)
        fen_rows.append(fen)

    try:
        return chess.Board("/".join(fen_rows) + " b - - 0 1")
    except:
        return None


print("\n=== YOLO Chess + Stockfish Started ===\n")

prev_time = 0
previous_board = None
countdown_start = None
state = "SHOW_BOARD"

try:
    stream = model.predict(source=WEBCAM_INDEX, stream=True,
                           conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)

    for result in stream:
        frame = result.orig_img
        if frame is None:
            continue

        frame = draw_boxes(frame, result.boxes, model.model.names)

        detected_board = detections_to_board(result, frame.shape)

        # ---------- STATE MACHINE ----------

        # 1️⃣ Show first REAL detected board
        if state == "SHOW_BOARD" and detected_board and detected_board.piece_map():
            previous_board = detected_board
            print("\nCurrent Board:")
            print(previous_board)
            print("\nWaiting for White to play...\n")
            state = "WAIT_WHITE"

        # 2️⃣ Wait for White move
        elif state == "WAIT_WHITE":
            if detected_board and detected_board.board_fen() != previous_board.board_fen():
                previous_board = detected_board
                countdown_start = time.time()
                print("White moved! Starting 20-second countdown...\n")
                state = "COUNTDOWN"

        # 3️⃣ Countdown in terminal
        elif state == "COUNTDOWN":
            remaining = int(WHITE_WAIT_TIME - (time.time() - countdown_start))

            if remaining >= 0:
                print(f"\rBlack thinking in: {remaining:2d}s", end="")
                time.sleep(1)
            else:
                print("\n\nCalculating Stockfish move...")
                try:
                    move = engine.play(previous_board, chess.engine.Limit(time=ENGINE_TIME)).move
                    print(f"\nBlack plays: {move}\n")
                except:
                    print("\nStockfish error\n")

                print("Waiting for next White move...\n")
                state = "WAIT_WHITE"

        # ---------- FPS only on video ----------
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO Chess Camera", frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

finally:
    cap.release()
    engine.quit()
    cv2.destroyAllWindows()
    print("\n=== Program Closed ===")
