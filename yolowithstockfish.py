import cv2
import numpy as np
import time
from ultralytics import YOLO
import chess
from stockfish import Stockfish


# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\best.pt"
STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-3\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
CONF_THRESHOLD = 0.387
IMG_SIZE = 864
width, height = 800, 800

# ---------------- CHESS ENGINE ----------------
DIFFICULTY_SETTINGS = {
    "easy": {"use_minimax": True},
    "medium": {"use_minimax": False, "Skill Level": 10, "Depth": 10},
    "hard": {"use_minimax": False, "Skill Level": 20, "Depth": 20},
}
stockfish = Stockfish(path=STOCKFISH_PATH)

def evaluate(board):
    piece_values = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 2000
    }
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

def minimax(board, depth, alpha, beta, is_maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

def find_best_move_minimax(board, depth=3):
    best_move = None
    # For Black (AI), we want to minimize the evaluation
    min_eval = float('inf')
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, float('-inf'), float('inf'), True)
        board.pop()
        if eval < min_eval:
            min_eval = eval
            best_move = move
    return best_move

def find_best_move_stockfish(board, difficulty):
    stockfish.set_fen_position(board.fen())
    stockfish.update_engine_parameters({"Skill Level": DIFFICULTY_SETTINGS[difficulty]["Skill Level"]})
    stockfish.set_depth(DIFFICULTY_SETTINGS[difficulty]["Depth"])
    move_uci = stockfish.get_best_move()
    return chess.Move.from_uci(move_uci) if move_uci else None

def find_best_move(board, difficulty):
    if DIFFICULTY_SETTINGS[difficulty].get("use_minimax"):
        return find_best_move_minimax(board)
    else:
        return find_best_move_stockfish(board, difficulty)

# ---------------- YOLO CONFIG ----------------
points, calibration_done, M, square_size = [], False, None, None
last_print_time = 0
piece_map = {
    "White-Pawn": "P", "White-Rook": "R", "White-Knight": "N",
    "White-Bishop": "B", "White-Queen": "Q", "White-King": "K",
    "Black-Pawn": "p", "Black-Rook": "r", "Black-Knight": "n",
    "Black-Bishop": "b", "Black-Queen": "q", "Black-King": "k",
}

def mouse_click(event, x, y, flags, param):
    global points, calibration_done, M, square_size
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")
    if len(points) == 4 and not calibration_done:
        pts = np.array(points, dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        ordered_pts = np.array([
            pts[np.argmin(s)], pts[np.argmin(diff)],
            pts[np.argmax(s)], pts[np.argmax(diff)]
        ], dtype="float32")
        dst_pts = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        square_size = width / 8
        calibration_done = True
        print("✅ Calibration complete.")

def map_point_to_square(x, y):
    global M, square_size
    pts = np.array([[[x, y]]], dtype="float32")
    mapped = cv2.perspectiveTransform(pts, M)[0][0]
    mx, my = mapped
    col, row = int(mx // square_size), 7 - int(my // square_size)
    return max(0, min(row, 7)), max(0, min(col, 7))

def board_matrix_to_fen(matrix, turn='w'):
    """Convert board matrix to FEN string with turn indicator."""
    fen_rows = []
    for row in matrix:
        fen_row, empty = "", 0
        for square in row:
            if square == ".":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += square
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + f" {turn} - - 0 1"

def fen_to_matrix(fen):
    """Convert FEN string into 8x8 board matrix."""
    fen_rows = fen.split(" ")[0].split("/")
    matrix = []
    for row in fen_rows:
        row_list = []
        for char in row:
            if char.isdigit():
                row_list.extend(["."] * int(char))
            else:
                row_list.append(char)
        matrix.append(row_list)
    return matrix

def print_matrix(matrix, label="Board Matrix"):
    """Print board matrix with proper formatting."""
    print(f"\n🔲 {label}:")
    print("  a b c d e f g h")
    for i, r in enumerate(matrix):
        print(f"{8-i} {' '.join(r)} {8-i}")
    print("  a b c d e f g h")

def draw_label(img, text, pos, bg_color=(0, 255, 0)):
    """Draw text label with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = pos
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness)

# ---------------- MAIN ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Chessboard Live")
cv2.setMouseCallback("Chessboard Live", mouse_click)
board = chess.Board()

difficulty = input("Choose AI difficulty (easy, medium, hard): ").strip().lower()
while difficulty not in DIFFICULTY_SETTINGS:
    difficulty = input("Invalid. Choose AI difficulty (easy, medium, hard): ").strip().lower()

print("👉 Click 4 corners of the chessboard (top-left, top-right, bottom-right, bottom-left).")
print("⚪ You are playing as WHITE (makes first move)")
print("⚫ AI (Stockfish) is playing as BLACK")

# Detection tracking
detected_pieces_count = 0
previous_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position

while True:
    ret, frame = cap.read()
    if not ret: break
    display = frame.copy()

    for p in points:
        cv2.circle(display, p, 5, (0, 0, 255), -1)

    board_matrix = [["." for _ in range(8)] for _ in range(8)]
    detected_pieces_count = 0
    
    if calibration_done and M is not None:
        results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id, cls_name = int(box.cls[0]), model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                row, col = map_point_to_square(cx, cy)
                
                # Map piece to FEN notation
                piece_fen = piece_map.get(cls_name, ".")
                
                # Handle multiple detections on same square (keep highest confidence)
                if board_matrix[row][col] == ".":
                    board_matrix[row][col] = piece_fen
                    detected_pieces_count += 1
                
                # Draw bounding box
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label with piece name, confidence, and position
                label_text = f"{cls_name} {conf:.2f} [{chr(97+col)}{8-row}]"
                draw_label(display, label_text, (int(x1), int(y1)), bg_color=(0, 255, 0))

        # Display detection count and turn indicator on screen
        turn_indicator = "WHITE's turn" if board.turn == chess.WHITE else "BLACK's turn (AI)"
        info_text = f"Detected: {detected_pieces_count} | {turn_indicator}"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        current_time = time.time()
        if current_time - last_print_time >= 80:  # every 10 sec
            print("\n" + "="*60)
            print(f"🔍 YOLO Detection Summary: {detected_pieces_count} pieces detected")
            print_matrix(board_matrix, label="🎯 YOLO Detected -> Board Matrix")
            
            # Determine whose turn it is
            turn = 'w' if board.turn == chess.WHITE else 'b'
            fen = board_matrix_to_fen(board_matrix, turn)
            print(f"\n📋 Generated FEN: {fen}")
            print(f"🎮 Current turn: {'WHITE (You)' if board.turn == chess.WHITE else 'BLACK (AI)'}")
            
            try:
                # Validate FEN before setting
                temp_board = chess.Board()
                temp_board.set_fen(fen)
                
                # Check if the position has changed (indicating a move was made)
                current_detected_fen = board_matrix_to_fen(board_matrix, turn)
                
                # Compare just the piece positions (first part of FEN)
                current_position = current_detected_fen.split()[0]
                previous_position = previous_fen.split()[0]
                position_changed = (current_position != previous_position)
                
                print(f"Position changed: {position_changed}")
                print(f"Previous: {previous_position}")
                print(f"Current:  {current_position}")
                
                if position_changed:
                    # Update main board to detected position
                    board.set_fen(fen)
                    
                    # Print Human Matrix
                    human_matrix = fen_to_matrix(board.fen())
                    print_matrix(human_matrix, label="🧑 Human Move -> Chess Board State")
                    
                    # Count pieces on chess board
                    board_piece_count = sum(1 for sq in chess.SQUARES if board.piece_at(sq) is not None)
                    print(f"♟️  Chess board has {board_piece_count} pieces")
                    
                    if detected_pieces_count != board_piece_count:
                        print(f"⚠️  WARNING: Mismatch! YOLO detected {detected_pieces_count} but board has {board_piece_count}")

                    # Only let AI reply if it's Black's turn (after White moves)
                    if board.turn == chess.BLACK:
                        print("🤖 BLACK (AI) to move...")
                        move = find_best_move(board, difficulty)
                        if move:
                            move_text = f"AI Move: {board.san(move)}"
                            print(f"\n🤖 {move_text}")
                            board.push(move)

                            # Print AI Matrix
                            ai_matrix = fen_to_matrix(board.fen())
                            print_matrix(ai_matrix, label="🤖 AI Move -> Chess Board State")
                            
                            # Update previous FEN after AI move
                            previous_fen = board.fen()
                        else:
                            print("⚠️  No legal AI move available")
                    elif board.turn == chess.WHITE:
                        print("⏳ Waiting for WHITE to make a move...")
                        # Update previous FEN after White's move is detected
                        previous_fen = board.fen()
                else:
                    print("ℹ️  No position change detected - board same as before")

            except ValueError as e:
                print(f"❌ Invalid FEN position: {e}")
                print("   Check if all pieces are correctly detected")
            except Exception as e:
                print(f"❌ FEN update error: {e}")

            last_print_time = current_time

    cv2.imshow("Chessboard Live", display)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()