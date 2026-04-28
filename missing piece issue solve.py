import cv2
import numpy as np
import time
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\best.pt"
CONF_THRESHOLD = 0.387
IMG_SIZE = 640

WEBCAM_INDEX = 0
BOARD_SIZE = 800
width, height = 800, 800

rotation_state = 1 # only for ENGINE, not visualization

# ---------------------------------------------

points = []
calibration_done = False
M = None
square_size = None
last_print_time = 0
homography = None

# ---------------- PIECE MAPPING ----------------
piece_map = {
    "White-Pawn": "P", "White-Rook": "R", "White-Knight": "N",
    "White-Bishop": "B", "White-Queen": "Q", "White-King": "K",
    "Black-Pawn": "p", "Black-Rook": "r", "Black-Knight": "n",
    "Black-Bishop": "b", "Black-Queen": "q", "Black-King": "k",
}

# ---------------- HELPERS ----------------
def matrix_to_fen(board):
    fen = ""
    for row in board:
        empty = 0
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
        fen += "/"
    return fen[:-1]

def rotate_board(board, k):
    return np.rot90(np.array(board), k=k).tolist()

def print_board(board):
    print("\nCurrent Board:")
    for i in range(8):
        rank = 8 - i
        print(rank, " ".join(board[i]))
    print("  a b c d e f g h\n")

# ---------------- CALIBRATION ----------------
def mouse_click(event, x, y, flags, param):
    global points, calibration_done, M, square_size, homography

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Corner {len(points)}: {x,y}")

    if len(points) == 4 and not calibration_done:
        pts = np.array(points, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")

        dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)

        dst_h = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype="float32")
        homography, _ = cv2.findHomography(ordered, dst_h)

        square_size = width / 8
        calibration_done = True
        print("✅ Calibration complete.")

# ---------------- MAP POINT ----------------
def map_point_to_square(x, y):
    pt = np.array([[[x,y]]], dtype="float32")
    mapped = cv2.perspectiveTransform(pt, M)[0][0]
    mx,my = mapped

    if mx < 0 or my < 0 or mx > width or my > height:
        return None

    col = int(mx // square_size)
    row = 7 - int(my // square_size)

    return max(0,min(7,row)), max(0,min(7,col))

# ---------------- DRAW WARPED GRID ----------------
def draw_fen_grid(img):
    step = BOARD_SIZE // 8
    for i in range(9):
        cv2.line(img,(0,i*step),(BOARD_SIZE,i*step),(0,255,0),1)
        cv2.line(img,(i*step,0),(i*step,BOARD_SIZE),(0,255,0),1)
    return img

# ---------------- INIT ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(WEBCAM_INDEX)
cv2.namedWindow("Chessboard Live")
cv2.setMouseCallback("Chessboard Live", mouse_click)

print("Click 4 board corners clockwise.")

# ---------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    board_matrix = [["." for _ in range(8)] for _ in range(8)]
    square_boxes = [[None for _ in range(8)] for _ in range(8)]

    if calibration_done:
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls[0])]
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                cx,cy = (x1+x2)/2, (y1+y2)/2

                pos = map_point_to_square(cx,cy)
                if pos is None:
                    continue

                row,col = pos
                area = (x2-x1)*(y2-y1)

                prev = square_boxes[row][col]
                if prev is None or area > prev[0]:
                    square_boxes[row][col] = (area, piece_map.get(cls,"."))

        # build board
        for r in range(8):
            for c in range(8):
                if square_boxes[r][c]:
                    board_matrix[r][c] = square_boxes[r][c][1]

        # -------- ENGINE TRUTH --------
        rotated_board = rotate_board(board_matrix, rotation_state)
        fen_position = matrix_to_fen(rotated_board)
        fen_string = fen_position + " w - - 0 1"

        # -------- PRINT TO TERMINAL --------
        if time.time() - last_print_time > 5:
            print("FEN:", fen_string)
            print_board(rotated_board)
            last_print_time = time.time()

        # -------- PROJECT BACK TO CAMERA --------
        invM = np.linalg.inv(M)

        for r in range(8):
            for c in range(8):
                pts = np.array([
                    [[c*square_size, r*square_size]],
                    [[(c+1)*square_size, r*square_size]],
                    [[(c+1)*square_size, (r+1)*square_size]],
                    [[c*square_size, (r+1)*square_size]],
                ], dtype="float32")

                real_pts = cv2.perspectiveTransform(pts, invM)
                real_pts = real_pts.reshape(-1,2).astype(int)

                cv2.polylines(display,[real_pts],True,(0,255,255),2)

                piece = rotated_board[r][c]
                if piece != ".":
                    cx = int(real_pts[:,0].mean())
                    cy = int(real_pts[:,1].mean())
                    cv2.putText(display,piece,(cx-10,cy+10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        # -------- WARPED ENGINE VIEW --------
        warped = cv2.warpPerspective(frame, homography, (BOARD_SIZE,BOARD_SIZE))
        warped = draw_fen_grid(warped)

        cv2.rectangle(warped,(0,0),(BOARD_SIZE,40),(0,0,0),-1)
        cv2.putText(warped,fen_string,(10,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow("Warped Board + FEN", warped)

    cv2.imshow("Chessboard Live", display)
    if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
        break

cap.release()
cv2.destroyAllWindows()
