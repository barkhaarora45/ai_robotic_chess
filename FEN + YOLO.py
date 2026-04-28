import chess
import chess.engine
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# ============ CHESS TRACKER CLASS ============

class LiveChessTracker:
    def __init__(self, stockfish_path=None):
        self.board = chess.Board()
        self.last_fen = self.board.fen()
        self.move_history = []
        self.stockfish_path = stockfish_path
        self.stable_fen_count = 0
        self.stable_threshold = 2
        self.pending_fen = None
        self.last_detected_fen = None
        self.auto_suggest = True
        self.calibration_fen = None  # Store calibrated starting position
        self.is_calibrated = False
        
    def calibrate_board(self, detected_fen):
        """Calibrate the tracker to the current board setup"""
        self.calibration_fen = detected_fen
        self.board = chess.Board(detected_fen.split(' ')[0] + ' w KQkq - 0 1')
        self.last_fen = self.board.fen()
        self.is_calibrated = True
        print(f"\n{'='*60}")
        print(f"✅ BOARD CALIBRATED!")
        print(f"{'='*60}")
        print(f"Calibration FEN: {self.calibration_fen}")
        print(f"\nBoard state recognized:")
        self._print_board_from_fen(self.board.fen())
        print(f"{'='*60}\n")
        
    def process_frame_fen(self, detected_fen):
        """Process FEN with stability check"""
        if not detected_fen:
            return None
            
        self.last_detected_fen = detected_fen
        
        if detected_fen == self.pending_fen:
            self.stable_fen_count += 1
        else:
            self.pending_fen = detected_fen
            self.stable_fen_count = 1
            
        if self.stable_fen_count >= self.stable_threshold:
            if detected_fen != self.last_fen:
                return self._detect_move(detected_fen)
        
        return None
    
    def _detect_move(self, detected_fen):
        """Detect and validate move"""
        legal_moves = []
        
        # Try all legal moves and see which one matches the detected position
        for move in self.board.legal_moves:
            self.board.push(move)
            diff = self._compare_fen(self.board.fen(), detected_fen)
            if diff <= 4:
                legal_moves.append((move, diff))
            self.board.pop()
        
        if legal_moves:
            legal_moves.sort(key=lambda x: x[1])
            best_move, diff = legal_moves[0]
            
            self.board.push(best_move)
            self.last_fen = self.board.fen()
            self.move_history.append(best_move)
            self.stable_fen_count = 0
            self.pending_fen = None
            
            return {
                'move': best_move,
                'uci': best_move.uci(),
                'san': self.board.san(best_move),
                'turn': 'White' if self.board.turn else 'Black',
                'move_number': len(self.move_history),
                'diff': diff
            }
        
        return None
    
    def _print_board_from_fen(self, fen):
        """Print a visual representation of the board from FEN"""
        board_part = fen.split()[0]
        rows = board_part.split('/')
        
        print("  +------------------------+")
        for i, row in enumerate(rows):
            expanded = ""
            for c in row:
                if c.isdigit():
                    expanded += '.' * int(c)
                else:
                    expanded += c
            
            row_str = f"{8-i} | "
            for piece in expanded:
                row_str += piece + " "
            row_str += "|"
            print(row_str)
        print("  +------------------------+")
        print("    a b c d e f g h")
    
    def _compare_fen(self, fen1, fen2):
        """Compare FEN positions - returns number of differences"""
        board1 = fen1.split()[0].replace('/', '')
        board2 = fen2.split()[0].replace('/', '')
        
        def expand(s):
            result = ""
            for c in s:
                result += '.' * int(c) if c.isdigit() else c
            return result
        
        exp1, exp2 = expand(board1), expand(board2)
        
        min_len = min(len(exp1), len(exp2))
        return sum(1 for a, b in zip(exp1[:min_len], exp2[:min_len]) if a != b)
    
    def get_stockfish_suggestion(self, for_black=True):
        """Get best move from Stockfish (default: for Black only)"""
        if not self.stockfish_path:
            return None
        
        if for_black and self.board.turn:
            return {'move': None, 'score': None, 'message': "⚪ It's White's turn - no suggestion"}
            
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            result = engine.analyse(self.board, chess.engine.Limit(time=0.5))
            best = result.get('pv', [None])[0]
            score = result.get('score')
            engine.quit()
            
            return {'move': best, 'score': score, 'message': None}
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None


# ============ YOLO TO FEN CONVERTER WITH AUTO-ORIENTATION ============

def yolo_detections_to_fen(boxes, names, orientation=0, debug=False):
    """
    Convert YOLO detections to FEN string
    orientation: 0=normal, 1=90° CW, 2=180°, 3=270° CW
    """
    if boxes is None or len(boxes) == 0:
        return None
    
    # Create empty 8x8 board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Get detections
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    
    # Map YOLO class names to FEN notation
    piece_map = {
        'Black-Bishop': 'b',
        'Black-King': 'k',
        'Black-Knight': 'n',
        'Black-Pawn': 'p',
        'Black-Queen': 'q',
        'Black-Rook': 'r',
        'White-Bishop': 'B',
        'White-King': 'K',
        'White-Knight': 'N',
        'White-Pawn': 'P',
        'White-Queen': 'Q',
        'White-Rook': 'R'
    }
    
    # Collect all pieces with their positions
    pieces = []
    for (x1, y1, x2, y2), cls, conf in zip(xyxy, clss, confs):
        piece_name = names.get(cls, '')
        if piece_name not in piece_map:
            continue
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        pieces.append({
            'name': piece_name,
            'fen': piece_map[piece_name],
            'x': center_x,
            'y': center_y,
            'conf': conf
        })
    
    if len(pieces) == 0:
        return None
    
    if debug:
        print(f"\n   Detected {len(pieces)} pieces")
    
    # Find board boundaries
    all_x = [p['x'] for p in pieces]
    all_y = [p['y'] for p in pieces]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Calculate square size
    square_width = (max_x - min_x) / 7
    square_height = (max_y - min_y) / 7
    
    # Place pieces on board
    for piece in pieces:
        col = int((piece['x'] - min_x) / square_width)
        row = int((piece['y'] - min_y) / square_height)
        
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        
        # Apply orientation
        if orientation == 1:  # 90° CW
            row, col = col, 7 - row
        elif orientation == 2:  # 180°
            row, col = 7 - row, 7 - col
        elif orientation == 3:  # 270° CW
            row, col = 7 - col, row
        
        if board[row][col] == '.':
            board[row][col] = piece['fen']
    
    # Convert board to FEN notation
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for square in row:
            if square == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    return fen


def find_best_orientation(boxes, names):
    """Try all 4 orientations and find which matches starting position best"""
    standard_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    
    best_orientation = 0
    best_diff = 999
    
    for orientation in range(4):
        fen = yolo_detections_to_fen(boxes, names, orientation=orientation)
        if fen:
            detected_board = fen.split()[0]
            diff = sum(1 for a, b in zip(detected_board.replace('/', ''), 
                                        standard_fen.replace('/', '')) if a != b)
            
            if diff < best_diff:
                best_diff = diff
                best_orientation = orientation
    
    return best_orientation, best_diff


# ============ OVERLAY DRAWING ============

def draw_chess_overlay(frame, tracker, move_info=None, stability=0, fps=0, orientation=0):
    """Draw chess information overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (10, 10), (450, 420), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 40
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 30
    
    # Show calibration status
    if tracker.is_calibrated:
        cv2.putText(frame, "Status: CALIBRATED ✓", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Status: NOT CALIBRATED - Press C", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    y_offset += 30
    
    # Show orientation
    orient_names = ["0°", "90° CW", "180°", "270° CW"]
    cv2.putText(frame, f"Orientation: {orient_names[orientation]}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 25
    
    if move_info:
        move_text = f"Last Move: {move_info['san']} ({move_info['uci']})"
        cv2.putText(frame, move_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        if 'diff' in move_info:
            cv2.putText(frame, f"Match Quality: {move_info['diff']} diffs", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 25
    
    turn = "White" if tracker.board.turn else "Black"
    color = (255, 255, 255) if tracker.board.turn else (150, 255, 150)
    cv2.putText(frame, f"Turn: {turn}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_offset += 35
    
    cv2.putText(frame, f"Moves: {len(tracker.move_history)}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 35
    
    stability_pct = (stability / tracker.stable_threshold) * 100
    color = (0, 255, 0) if stability >= tracker.stable_threshold else (0, 165, 255)
    cv2.putText(frame, f"Stability: {stability_pct:.0f}%", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    bar_width = int(350 * stability_pct / 100)
    cv2.rectangle(frame, (20, y_offset + 10), (370, y_offset + 25), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, y_offset + 10), (20 + bar_width, y_offset + 25), color, -1)
    y_offset += 45
    
    if not tracker.board.turn and tracker.is_calibrated:
        cv2.putText(frame, "⚫ WAITING FOR BLACK MOVE...", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        y_offset += 30
    
    cv2.putText(frame, "Recent Moves:", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    for i, move in enumerate(tracker.move_history[-3:]):
        move_num = len(tracker.move_history) - 2 + i
        if move_num > 0:
            cv2.putText(frame, f"{move_num}. {move.uci()}", (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 20
    
    cv2.putText(frame, "Q-Quit | C-Calibrate | O-Rotate | S-Hint", (20, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame


def draw_yolo_boxes(frame, boxes, names):
    """Draw YOLO detection boxes"""
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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return frame


# ============ MAIN PROGRAM ============

def main():
    MODEL_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\best.pt"
    STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    WEBCAM_INDEX = 0
    CONF_THRESHOLD = 0.398
    IMG_SIZE = 672
    
    print("=" * 60)
    print("♟️  LIVE CHESS TRACKER WITH AUTO-CALIBRATION")
    print("=" * 60)
    print(f"Loading YOLO model: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    tracker = LiveChessTracker(stockfish_path=STOCKFISH_PATH)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🎯 SETUP INSTRUCTIONS:")
    print("=" * 60)
    print("1. Place your chessboard in starting position")
    print("2. Position camera so all pieces are visible")
    print("3. Press 'C' to CALIBRATE (learns your setup)")
    print("4. If orientation is wrong, press 'O' to rotate")
    print("5. Once calibrated, start playing!")
    print("=" * 60)
    
    print("\nControls:")
    print("  C - Calibrate board (IMPORTANT: do this first!)")
    print("  O - Cycle through orientations (0°, 90°, 180°, 270°)")
    print("  S - Get Stockfish suggestion (BLACK ONLY)")
    print("  R - Reset game")
    print("  Q - Quit")
    print("\n⚫ STOCKFISH IS PLAYING AS BLACK")
    print("💡 Auto-suggest: ON (will suggest after White moves)")
    print("=" * 60)
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    current_move_info = None
    prev_time = 0.0
    frame_count = 0
    orientation = 0
    last_auto_suggest_move = -1
    
    try:
        stream = model.predict(source=WEBCAM_INDEX, stream=True, 
                              conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        
        for result in stream:
            frame = result.orig_img
            if frame is None:
                continue
            
            frame_count += 1
            
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
            prev_time = cur_time
            
            frame = draw_yolo_boxes(frame, result.boxes, model.names)
            
            detected_fen = yolo_detections_to_fen(result.boxes, model.names, 
                                                  orientation=orientation)
            
            # Only process moves if calibrated
            if tracker.is_calibrated and detected_fen:
                move_info = tracker.process_frame_fen(detected_fen)
                
                if move_info:
                    current_move_info = move_info
                    print(f"\n{'='*60}")
                    print(f"🎯 MOVE DETECTED!")
                    print(f"{'='*60}")
                    print(f"  Move: {move_info['san']} ({move_info['uci']})")
                    print(f"  Move Number: {move_info['move_number']}")
                    print(f"  Next Turn: {move_info['turn']}")
                    print(f"  Match Quality: {move_info['diff']} differences")
                    print(f"  FEN: {tracker.board.fen()}")
                    print(f"{'='*60}")
                    
                    if tracker.auto_suggest and not tracker.board.turn and \
                       len(tracker.move_history) != last_auto_suggest_move:
                        last_auto_suggest_move = len(tracker.move_history)
                        print("\n⏳ Auto-suggesting move for Black...")
                        time.sleep(0.5)
                        suggestion = tracker.get_stockfish_suggestion(for_black=True)
                        if suggestion and suggestion['move']:
                            print(f"\n{'='*60}")
                            print(f"⚫ STOCKFISH SUGGESTS FOR BLACK:")
                            print(f"{'='*60}")
                            print(f"  Move: {suggestion['move']}")
                            if suggestion['score']:
                                print(f"  Evaluation: {suggestion['score']}")
                            print(f"{'='*60}")
                            print(f"\n👉 Make this move on the board!")
            
            frame = draw_chess_overlay(frame, tracker, current_move_info, 
                                      tracker.stable_fen_count, fps, orientation)
            
            cv2.imshow("Live Chess Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n👋 Quitting...")
                break
            
            elif key == ord('c') or key == ord('C'):
                if detected_fen:
                    tracker.calibrate_board(detected_fen)
                else:
                    print("\n❌ Cannot calibrate - no pieces detected!")
                    print("   Make sure all pieces are visible to camera")
            
            elif key == ord('o') or key == ord('O'):
                orientation = (orientation + 1) % 4
                orient_names = ["0° (normal)", "90° clockwise", "180° (flipped)", "270° clockwise"]
                print(f"\n🔄 Orientation changed to: {orient_names[orientation]}")
                
            elif key == ord('r') or key == ord('R'):
                tracker = LiveChessTracker(stockfish_path=STOCKFISH_PATH)
                current_move_info = None
                last_auto_suggest_move = -1
                print("\n♟️  GAME RESET! Press 'C' to recalibrate.")
                
            elif key == ord('s') or key == ord('S'):
                if not tracker.is_calibrated:
                    print("\n⚠️  Please calibrate first (Press 'C')")
                else:
                    print("\n🤔 Thinking...")
                    suggestion = tracker.get_stockfish_suggestion(for_black=True)
                    if suggestion:
                        if suggestion.get('message'):
                            print(f"ℹ️  {suggestion['message']}")
                        elif suggestion['move']:
                            print(f"⚫ Stockfish suggests for BLACK: {suggestion['move']}")
                            if suggestion['score']:
                                print(f"   Evaluation: {suggestion['score']}")
                        else:
                            print("❌ No suggestion available")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Chess Tracker closed!")


if __name__ == "__main__":
    main()