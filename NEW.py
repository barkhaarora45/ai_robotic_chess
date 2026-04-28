import cv2
import numpy as np
import chess

"""
Interactive 2D board calibration tool with FEN grid preview
-----------------------------------------------------------
Workflow:
1. Live camera opens.
2. User clicks FOUR board corners (clockwise).
3. Press 'K' to confirm.
4. Shows warped 2D board with **8×8 FEN label grid overlay**.
"""

# -------- USER CONFIG --------
WEBCAM_INDEX = 0
BOARD_SIZE = 800  # size of warped square board
# -----------------------------

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

clicked_points = []
rotation_state = 1  # 🔒 LOCKED at 90°
calibrated = False
homography = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Corner {len(clicked_points)}: ({x}, {y})")


cv2.namedWindow("Warped Board + FEN")
cv2.setMouseCallback("Warped Board + FEN", mouse_callback)

print("""
=== 2D Chess Board Calibration ===

Steps:
1. Click FOUR board corners clockwise.
2. Press 'K' to confirm calibration.
Press ESC to exit.
""")


def draw_fen_grid(board_img):
    """Draw 8x8 grid with chess square labels (a1–h8)."""
    step = BOARD_SIZE // 8

    # Draw grid lines
    for i in range(9):
        cv2.line(board_img, (0, i * step), (BOARD_SIZE, i * step), (0, 255, 0), 1)
        cv2.line(board_img, (i * step, 0), (i * step, BOARD_SIZE), (0, 255, 0), 1)

    # Draw FEN square labels
    files = "abcdefgh"

    for r in range(8):
        for c in range(8):
            square = f"{files[c]}{8 - r}"

            text_x = int(c * step + step * 0.3)
            text_y = int(r * step + step * 0.7)

            cv2.putText(
                board_img,
                square,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return board_img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- BEFORE 4 CLICKS → SHOW LIVE CAMERA ----------
    if len(clicked_points) < 4:
        cv2.imshow("Warped Board + FEN", frame)

    # ---------- AFTER 4 CLICKS → SHOW WARPED BOARD ----------
    else:
        # Compute homography once
        if homography is None:
            dst = np.array([
                [0, 0],
                [BOARD_SIZE - 1, 0],
                [BOARD_SIZE - 1, BOARD_SIZE - 1],
                [0, BOARD_SIZE - 1]
            ], dtype=np.float32)

            src = np.array(clicked_points, dtype=np.float32)
            homography, _ = cv2.findHomography(src, dst)

        # Warp to top-down board
        warped = cv2.warpPerspective(frame, homography, (BOARD_SIZE, BOARD_SIZE))

        # 🔒 Always rotate 90°
        rotated = np.rot90(warped, k=rotation_state).copy()

        # Draw FEN grid
        rotated = draw_fen_grid(rotated)

        # Placeholder FEN (until YOLO added)
        fen_string = "8/8/8/8/8/8/8/8 w - - 0 1"

        # Draw FEN text bar
        cv2.rectangle(rotated, (0, 0), (BOARD_SIZE, 40), (0, 0, 0), -1)
        cv2.putText(
            rotated,
            fen_string,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Warped Board + FEN", rotated)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key in [ord('k'), ord('K')] and len(clicked_points) == 4:
        calibrated = True
        print("Calibration confirmed!")
        break


cap.release()
cv2.destroyAllWindows()


if not calibrated:
    print("Calibration cancelled.")
    exit()


# -------- Generate square centers for FEN mapping --------
print("\nGenerating 8x8 square centers for FEN mapping...")

square_centers = []
step = BOARD_SIZE // 8

for r in range(8):
    row = []
    for c in range(8):
        x = int((c + 0.5) * step)
        y = int((r + 0.5) * step)
        row.append((x, y))
    square_centers.append(row)

print("Square centers (in warped board coordinates):")
for row in square_centers:
    print(row)

print("""
Calibration complete.

Next step:
➡ Feed YOLO detections through this homography
➡ Map to nearest square center
➡ Build accurate FEN for Stockfish
""")
