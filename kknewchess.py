import cv2
import numpy as np
import time
from ultralytics import YOLO
import chess
from stockfish import Stockfish

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\best.pt"
STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

CONF_THRESHOLD = 0.45
IMG_SIZE = 864
width, height = 800, 800
WHITE_WAIT_TIME = 20

# ---------------- STOCKFISH ----------------
stockfish = Stockfish(path=STOCKFISH_PATH)
stockfish.update_engine_parameters({"Skill Level": 15})
stockfish.set_depth(15)

# ---------------- YOLO STATE ----------------
points, calibration_done, M, square_size = [], False, None, None
last_fen = None
initial_board_printed = False
waiting_for_black_sync = False

piece_map = {
    "White-Pawn": "P", "White-Rook": "R", "White-Knight": "N",
    "White-Bishop": "B", "White-Queen": "Q", "White-King": "K",
    "Black-Pawn": "p", "Black-Rook": "r", "Black-Knight": "n",
    "Black-Bishop": "b", "Black-Queen": "q", "Black-King": "k",
}

# ---------------- CALIBRATION ----------------
def mouse_click(event, x, y, flags, param):
    global points, calibration_done, M, square_size

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")

    if len(points) == 4 and not calibration_done:
        pts = np.array(points, dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)

        ordered = np.array([
            pts[np.argmin(s)],      # top-left
            pts[np.argmin(diff)],   # top-right
            pts[np.argmax(s)],      # bottom-right
            pts[np.argmax(diff)]    # bottom-left
        ], dtype="float32")

        dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        square_size = width / 8
        calibration_done = True
        print("\n✅ Calibration complete.\n")

# ---------------- FIXED 180° ORIENTATION ----------------
def map_point_to_square(x, y):
    pts = np.array([[[x, y]]], dtype="float32")
    mx, my = cv2.perspectiveTransform(pts, M)[0][0]

    # Rotate board 180° to match python-chess orientation
    col = 7 - int(mx // square_size)
    row = int(my // square_size)

    return max(0, min(row,7)), max(0, min(col,7))

# ---------------- FEN HELPERS ----------------
def board_matrix_to_fen(matrix, turn):
    fen_rows = []
    for row in matrix:
        fen_row, empty = "", 0
        for sq in row:
            if sq == ".":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += sq
        if empty:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    return "/".join(fen_rows) + f" {turn} - - 0 1"

def fen_to_matrix(fen):
    rows = fen.split(" ")[0].split("/")
    matrix = []
    for r in rows:
        row = []
        for ch in r:
            if ch.isdigit():
                row.extend(["."] * int(ch))
            else:
                row.append(ch)
        matrix.append(row)
    return matrix

def print_matrix(matrix, label):
    print(f"\n🔲 {label}:")
    for r in matrix:
        print(" ".join(r))

# ---------------- DETECTION ----------------
def detect_board(frame):
    board_matrix = [["." for _ in range(8)] for _ in range(8)]
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1+x2)/2, (y1+y2)/2
            row, col = map_point_to_square(cx, cy)
            board_matrix[row][col] = piece_map.get(cls, ".")

    return board_matrix

# ---------------- INIT ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Chessboard Live")
cv2.setMouseCallback("Chessboard Live", mouse_click)

board = chess.Board()

print("👉 Click 4 corners to calibrate.\n")

# =========================================================
# ====================== MAIN LOOP ========================
# =========================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Chessboard Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if not calibration_done:
        continue

    # -------- INITIAL BOARD PRINT --------
    if not initial_board_printed:
        board_matrix = detect_board(frame)
        fen_start = board_matrix_to_fen(board_matrix, turn="w")

        try:
            board.set_fen(fen_start)
            print_matrix(fen_to_matrix(board.fen()), "📍 Current Board Before White Move")
            last_fen = board.fen()
            initial_board_printed = True
            print("\n⏳ Waiting for WHITE move...\n")
        except:
            continue

    # -------- WAIT FOR BLACK SYNC --------
    if waiting_for_black_sync:
        board_matrix = detect_board(frame)
        fen_camera = board_matrix_to_fen(board_matrix, turn="w")

        if fen_camera == last_fen:
            print("✅ Black move detected on physical board.")
            waiting_for_black_sync = False
            print("\n⏳ Waiting for WHITE move...\n")
            time.sleep(2)

        continue

    # -------- WHITE TURN --------
    print("⏳ White thinking...")
    for i in range(WHITE_WAIT_TIME, 0, -1):
        print(f"   {i:02d}s remaining", end="\r")
        time.sleep(1)
    print(" " * 30, end="\r")

    ret, frame = cap.read()
    if not ret:
        continue

    board_matrix = detect_board(frame)
    fen_white = board_matrix_to_fen(board_matrix, turn="b")

    if fen_white == last_fen:
        print("⚠️ No move detected from White.")
        continue

    last_fen = fen_white

    try:
        board.set_fen(fen_white)
        print_matrix(fen_to_matrix(board.fen()), "🧑 WHITE played")

        print("\n🤖 Stockfish thinking...")
        stockfish.set_fen_position(board.fen())
        move_uci = stockfish.get_best_move()

        if move_uci:
            move = chess.Move.from_uci(move_uci)
            print(f"🤖 BLACK plays: {move_uci} | SAN: {board.san(move)}")

            board.push(move)
            print_matrix(fen_to_matrix(board.fen()), "🤖 BLACK board")

            last_fen = board.fen()
            waiting_for_black_sync = True
            print("\n🔄 Waiting for Black move to appear on board...")

    except Exception as e:
        print("❌ Detection error:", e)

cap.release()
cv2.destroyAllWindows()