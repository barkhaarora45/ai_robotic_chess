import chess
from stockfish import Stockfish

# Path to your Stockfish binary
STOCKFISH_PATH = r"C:\Users\Sanjhvi\Desktop\RossChess-2\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # Example: "C:/stockfish/stockfish.exe"

# Difficulty levels configuration
DIFFICULTY_SETTINGS = {
    "easy": {"use_minimax": True},
    "medium": {"use_minimax": False, "Skill Level": 10, "Depth": 10},
    "hard": {"use_minimax": False, "Skill Level": 20, "Depth": 20},
}

# Initialize Stockfish engine
stockfish = Stockfish(path=STOCKFISH_PATH)

# Evaluation function for minimax
def evaluate(board):
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 2000
    }
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

# Minimax with alpha-beta pruning
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
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Minimax-based move finder (easy mode)
def find_best_move_minimax(board, depth):
    best_move = None
    max_eval = float('-inf')
    legal_moves = list(board.legal_moves)

    def order_moves(move):
        return 1 if board.is_capture(move) else 0

    legal_moves.sort(key=order_moves, reverse=True)

    for move in legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, float('-inf'), float('inf'), False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            best_move = move
    return best_move

# Stockfish-based move finder (medium, hard)
def find_best_move_stockfish(board, difficulty):
    stockfish.set_fen_position(board.fen())
    stockfish.update_engine_parameters({
        "Skill Level": DIFFICULTY_SETTINGS[difficulty]["Skill Level"]
    })
    stockfish.set_depth(DIFFICULTY_SETTINGS[difficulty]["Depth"])
    move_uci = stockfish.get_best_move()
    return chess.Move.from_uci(move_uci) if move_uci else None

# Main move finder
def find_best_move(board, difficulty):
    if DIFFICULTY_SETTINGS[difficulty].get("use_minimax"):
        return find_best_move_minimax(board, depth=3)
    else:
        return find_best_move_stockfish(board, difficulty)

# Game loop
def play_chess():
    board = chess.Board()
    difficulty = ""
    
    while difficulty not in DIFFICULTY_SETTINGS:
        difficulty = input("Choose AI difficulty (easy, medium, hard): ").strip().lower()

    while not board.is_game_over():
        print(board, "\n")

        if board.turn == chess.WHITE:
            move_input = input("Enter your move: ")
            try:
                board.push_san(move_input)
            except:
                print("Invalid move, try again.")
                continue
        else:
            print(f"AI ({difficulty}) is thinking...")
            move = find_best_move(board, difficulty)
            if move:
                move_san = board.san(move)  # Get SAN before pushing
                board.push(move)
                print("AI played:", move_san)
            else:
                print("No move found by AI.")
        print()

    print(board)
    result = board.result()
    if result == '1-0':
        print("White wins!")
    elif result == '0-1':
        print("Black wins!")
    else:
        print("It's a draw!")

# Start the game
play_chess()