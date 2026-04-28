import cv2
import numpy as np
import time
from ultralytics import YOLO
import chess
from stockfish import Stockfish
import serial

# =========================================================
# ================= ROBOT ARM SECTION =====================
# =========================================================

PORT = "COM8"
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

STEP_DELAY = 0.08
ANGLE_STEP = 3

GRIP_OPEN = 0
GRIP_CLOSE = 60

HOME = [90, 180, 90, 90, GRIP_OPEN]

board_positions = {
    "a8": [58,  58, 114,90],
    "b8": [76,  76,  92,90],
    "c8": [80,  70,  94,90],
    "d8": [105, 84, 76,82],
    "e8": [120, 80, 87,89],
    "f8": [153, 53, 123,90],
    "g8": [155, 50, 130,89],
    "h8": [161,	35,	150,89],
    "h7": [148,	27,	169,90],
    "g7": [142,	33,	143,90],
    "f7": [134,	54,	127,90],
    "e7": [113,	61,	110,90],
    "d7": [98,	62,	108,90],
    "c7": [85,	60,	114,90],
    "b7": [70,	53,	125,90],
    "a7": [60,  47, 134,90],
    "a6": [64,  30, 166,90],
    "b6": [75,  38, 151,90],
    "c6": [87, 50, 134,90],
    "d6": [97, 49, 136,90],
    "e6": [109, 44, 143,90],
    "f6": [118, 38, 153,90],
    "g6": [130, 30, 166,90],
    "h6": [137, 20, 180,82],
    "h5": [132, 21, 180,72],
    "g5": [123, 24, 180,85],
    "f5": [115, 24, 180,91],
    "e5": [107, 36, 165,89],
    "d5": [98, 33, 165,89],
    "c5": [89, 27, 176,89],
    "b5": [81, 23, 180,87],
    "a5": [72, 23, 180,80],
    "a4": [78, 21, 180,63],
    "b4": [85, 22, 180,72],
    "c4": [92, 25, 180,78],
}

PIECE_SIZE = {"k":"big","q":"big","r":"small","p":"small","b":"medium","n":"medium"}
SIZE_OFFSETS = {"big":{},"small":{1:-10},"medium":{}}

def send(pose):
    ser.write((",".join(map(str, pose)) + "\n").encode())
    time.sleep(STEP_DELAY)

def move_joint(current, joint_index, target):
    start = current[joint_index]
    step = ANGLE_STEP if target > start else -ANGLE_STEP
    for angle in range(start, target, step):
        current[joint_index] = angle
        send(current)
    current[joint_index] = target
    send(current)
    return current

def adjusted_pose(square, piece_code):
    pose = board_positions[square].copy()
    pose.append(GRIP_OPEN)
    size = PIECE_SIZE.get(piece_code.lower(), "medium")
    for idx, delta in SIZE_OFFSETS[size].items():
        pose[idx] += delta
    return pose

def move_home_to_pick(current, pick):
    current = move_joint(current, 0, pick[0])
    current = move_joint(current, 2, pick[2])
    current = move_joint(current, 3, pick[3])
    current = move_joint(current, 1, pick[1])
    return current

def go_home(current):
    current = move_joint(current, 1, HOME[1])
    current = move_joint(current, 0, HOME[0])
    current = move_joint(current, 2, HOME[2])
    current = move_joint(current, 3, HOME[3])
    current[4] = GRIP_OPEN
    send(current)
    return current

def move_piece(src, dst, piece_code):
    print(f"🤖 ARM: {src} → {dst}")
    current = HOME.copy()
    send(current)
    time.sleep(0.5)

    pick_pose = adjusted_pose(src, piece_code)
    current = move_home_to_pick(current, pick_pose)

    current[4] = GRIP_CLOSE
    send(current)
    time.sleep(0.5)

    current = move_joint(current, 1, 138)

    place_pose = adjusted_pose(dst, piece_code)
    current = move_joint(current, 0, place_pose[0])
    current = move_joint(current, 2, place_pose[2])
    current = move_joint(current, 3, place_pose[3])
    current = move_joint(current, 1, place_pose[1])

    current[4] = GRIP_OPEN
    send(current)
    time.sleep(0.5)

    go_home(current)
    print("🤖 ARM Move complete\n")

# =========================================================
# ================= CHESS + YOLO SECTION ==================
# =========================================================

MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\best.pt"
STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

CONF_THRESHOLD = 0.387
IMG_SIZE = 864
width, height = 800, 800
WHITE_WAIT_TIME = 20

stockfish = Stockfish(path=STOCKFISH_PATH)
stockfish.update_engine_parameters({"Skill Level": 15})
stockfish.set_depth(15)

points, calibration_done, M, square_size = [], False, None, None
last_fen = None
initial_board_printed = False

piece_map = {
    "White-Pawn":"P","White-Rook":"R","White-Knight":"N",
    "White-Bishop":"B","White-Queen":"Q","White-King":"K",
    "Black-Pawn":"p","Black-Rook":"r","Black-Knight":"n",
    "Black-Bishop":"b","Black-Queen":"q","Black-King":"k",
}

def mouse_click(event,x,y,flags,param):
    global points, calibration_done, M, square_size
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x,y))
        print(f"Point: {x} {y}")
    if len(points) == 4 and not calibration_done:
        pts = np.array(points, dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")
        dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        square_size = width/8
        calibration_done = True
        print("✅ Calibration complete\n")

# 🔥 180° ROTATION FIX (Camera on BLACK side)
def map_point_to_square(x,y):
    pts = np.array([[[x,y]]], dtype="float32")
    mx,my = cv2.perspectiveTransform(pts, M)[0][0]
    col = 7 - int(mx // square_size)
    row = int(my // square_size)
    return max(0,min(row,7)), max(0,min(col,7))

def detect_board(frame):
    matrix = [["." for _ in range(8)] for _ in range(8)]
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            cx,cy = (x1+x2)/2, (y1+y2)/2
            row,col = map_point_to_square(cx,cy)
            matrix[row][col] = piece_map.get(cls,".")
    return matrix

def board_matrix_to_fen(matrix,turn):
    fen=[]
    for row in matrix:
        s=""; empty=0
        for sq in row:
            if sq==".":
                empty+=1
            else:
                if empty: s+=str(empty); empty=0
                s+=sq
        if empty: s+=str(empty)
        fen.append(s)
    return "/".join(fen)+f" {turn} - - 0 1"

def fen_to_matrix(fen):
    rows=fen.split(" ")[0].split("/")
    matrix=[]
    for r in rows:
        row=[]
        for ch in r:
            if ch.isdigit(): row.extend(["."]*int(ch))
            else: row.append(ch)
        matrix.append(row)
    return matrix

def print_matrix(matrix,label):
    print(f"\n🔲 {label}:")
    for r in matrix:
        print(" ".join(r))

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Chessboard Live")
cv2.setMouseCallback("Chessboard Live", mouse_click)

board = chess.Board()

print("👉 Click 4 corners to calibrate.\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Chessboard Live", frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break
    if not calibration_done: continue

    if not initial_board_printed:
        matrix = detect_board(frame)
        fen = board_matrix_to_fen(matrix,"w")
        board.set_fen(fen)
        print_matrix(fen_to_matrix(board.fen()),"📍 Current Board Before White Move")
        last_fen = board.fen()
        initial_board_printed = True
        print("\n⏳ Waiting for WHITE move...\n")

    print("⏳ White thinking...")
    for i in range(WHITE_WAIT_TIME,0,-1):
        print(f"   {i:02d}s remaining",end="\r")
        time.sleep(1)
    print(" "*30,end="\r")

    ret, frame = cap.read()
    if not ret: continue

    matrix = detect_board(frame)
    fen_white = board_matrix_to_fen(matrix,"b")

    if fen_white==last_fen:
        print("⚠️ No move detected from White.")
        continue

    board.set_fen(fen_white)
    print_matrix(fen_to_matrix(board.fen()),"🧑 WHITE played")
    last_fen = board.fen()

    print("\n🤖 Stockfish thinking...")
    stockfish.set_fen_position(board.fen())
    move_uci = stockfish.get_best_move()

    if move_uci:
        move = chess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        piece_code = piece.symbol()

        print(f"🤖 BLACK plays: {move_uci} | SAN: {board.san(move)}")

        board.push(move)
        print_matrix(fen_to_matrix(board.fen()),"🤖 BLACK board")

        src,dst = move_uci[:2], move_uci[2:4]
        move_piece(src,dst,piece_code)

        last_fen = board.fen()
        print("\n⏳ Waiting for WHITE move...\n")

cap.release()
cv2.destroyAllWindows()
ser.close()