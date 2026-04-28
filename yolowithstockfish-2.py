import time
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import chess
import chess.engine

# ---------------- USER CONFIG ----------------
# Path to your trained best weights (change to your path)
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\best.pt"   # <- change this to your best.pt path

# Webcam index (0 is default internal webcam). Or set to video file path.
WEBCAM_INDEX = 0

# Confidence threshold (use the value from your F1 curve; you said ~0.37)
CONF_THRESHOLD = 0.387

# Image size to send to YOLO (640 is fine)
IMG_SIZE = 864

# Whether to save the output video
SAVE_OUTPUT = False
OUT_VIDEO_PATH = "live_output.mp4"

# -------- STOCKFISH CONFIG (NEW) --------
# Path to Stockfish executable - CHANGE THIS to your actual Stockfish path
STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # Windows
# STOCKFISH_PATH = "/usr/local/bin/stockfish"  # Linux/Mac

# Stockfish skill level (0-20, where 20 is strongest)
STOCKFISH_SKILL_LEVEL = 10

# Time limit for Stockfish to think (in seconds)
STOCKFISH_TIME_LIMIT = 1.0

# Enable Stockfish integration
USE_STOCKFISH = True

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

# -------- STOCKFISH INITIALIZATION (NEW) --------
engine = None
board = None
if USE_STOCKFISH:
    if not os.path.exists(STOCKFISH_PATH):
        print(f"WARNING: Stockfish not found at {STOCKFISH_PATH}")
        print("Stockfish integration disabled. Please update STOCKFISH_PATH.")
        USE_STOCKFISH = False
    else:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            engine.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})
            board = chess.Board()
            print(f"Stockfish initialized at skill level {STOCKFISH_SKILL_LEVEL}")
            print("Stockfish will play as BLACK")
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            USE_STOCKFISH = False

# Mapping from YOLO class names to chess pieces
# Adjust these based on your actual YOLO class names
PIECE_MAPPING = {
    'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B',
    'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b',
    'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k'
}

def detect_board_state(boxes, names):
    """
    Convert YOLO detections to board state.
    This is a simplified version - you'll need to calibrate this based on your setup.
    
    Returns: 8x8 board array or None if detection fails
    """
    if boxes is None or len(boxes) == 0:
        return None
    
    # Get detections
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    
    # Extract piece positions and types
    pieces = []
    for (x1, y1, x2, y2), c in zip(xyxy, clss):
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        piece_name = names.get(c, str(c))
        pieces.append({
            'x': center_x,
            'y': center_y,
            'name': piece_name,
            'fen': PIECE_MAPPING.get(piece_name.lower(), None)
        })
    
    # TODO: Implement proper board position detection
    # This requires calibration to map pixel positions to chess squares
    # For now, return None to use manual move input
    return None

def get_stockfish_move(board, engine, time_limit=1.0):
    """Get Stockfish's suggested move for the current position."""
    if board.is_game_over():
        return None
    
    try:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    except Exception as e:
        print(f"Error getting Stockfish move: {e}")
        return None

def draw_stockfish_suggestion(frame, suggested_move, board_corners=None):
    """Draw Stockfish's suggested move on the frame."""
    if suggested_move is None:
        return frame
    
    # Display move in text format
    move_text = f"Stockfish suggests: {suggested_move.uci()}"
    cv2.putText(frame, move_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # TODO: Draw arrow on board if board corners are detected
    # This requires board corner detection and homography
    
    return frame

# ------------------------------------------------

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
if USE_STOCKFISH:
    print("Press 'm' to manually input a move")
    print("Press 's' to get Stockfish suggestion")
    print("Press 'r' to reset the board")

prev_time = 0.0
suggested_move = None

try:
    # Stream mode: yields results per frame (stream=True)
    # We pass conf (confidence), imgsz and device will be handled by ultralytics
    stream = model.predict(source=WEBCAM_INDEX, stream=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE , verbose =False)

    for result in stream:
        # result.orig_img is the original BGR frame (numpy)
        frame = result.orig_img  # already numpy BGR
        if frame is None:
            continue

        # Draw boxes manually (so we control font/color)
        frame = draw_boxes(frame, result.boxes, model.model.names)  # model.model.names or model.names works

        # -------- STOCKFISH INTEGRATION (NEW) --------
        if USE_STOCKFISH:
            # Display current board state
            if board is not None:
                turn_text = "White to move" if board.turn == chess.WHITE else "Black to move"
                cv2.putText(frame, turn_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display move count
                move_count_text = f"Move: {board.fullmove_number}"
                cv2.putText(frame, move_count_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw suggested move if available
                if suggested_move is not None:
                    frame = draw_stockfish_suggestion(frame, suggested_move)

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
        
        # -------- STOCKFISH KEY CONTROLS (NEW) --------
        elif USE_STOCKFISH and key == ord("m"):  # Manual move input
            print("\n" + "="*50)
            print("Current board:")
            print(board)
            print("="*50)
            move_input = input("Enter move in UCI format (e.g., e2e4) or SAN (e.g., Nf3): ").strip()
            try:
                # Try parsing as UCI first
                try:
                    move = chess.Move.from_uci(move_input)
                except:
                    # Try parsing as SAN
                    move = board.parse_san(move_input)
                
                if move in board.legal_moves:
                    board.push(move)
                    print(f"Move {move} applied.")
                    
                    # If it's now black's turn and Stockfish plays black, get suggestion
                    if board.turn == chess.BLACK and not board.is_game_over():
                        print("Getting Stockfish's move...")
                        suggested_move = get_stockfish_move(board, engine, STOCKFISH_TIME_LIMIT)
                        if suggested_move:
                            print(f"Stockfish suggests: {suggested_move.uci()}")
                            # Optionally auto-play the move
                            auto_play = input("Apply Stockfish's move? (y/n): ").strip().lower()
                            if auto_play == 'y':
                                board.push(suggested_move)
                                print(f"Stockfish played: {suggested_move.uci()}")
                                suggested_move = None
                    else:
                        suggested_move = None
                else:
                    print(f"Illegal move: {move_input}")
            except Exception as e:
                print(f"Invalid move format: {e}")
        
        elif USE_STOCKFISH and key == ord("s"):  # Get Stockfish suggestion
            if board is not None and not board.is_game_over():

                print("Getting Stockfish suggestion...")
                suggested_move = get_stockfish_move(board, engine, STOCKFISH_TIME_LIMIT)
                if suggested_move:
                    print(f"Stockfish suggests: {suggested_move.uci()} ({board.san(suggested_move)})")
        
        elif USE_STOCKFISH and key == ord("r"):  # Reset board
            board.reset()
            suggested_move = None
            print("Board reset to starting position")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # -------- CLEANUP STOCKFISH (NEW) --------
    if engine is not None:
        engine.quit()
        print("Stockfish engine closed.")
    
    print("Exiting.")