import chess
import chess.engine

def find_best_matching_fen(detected_fen, stockfish_path, max_differences=3):
    """
    Match detected FEN with legal positions using Stockfish
    """
    try:
        board = chess.Board(detected_fen)
        
        # If FEN is valid and legal, return it
        if board.is_valid():
            return detected_fen, 0
    except:
        pass
    
    # If invalid, find closest legal position
    return None, -1


def match_with_game_state(detected_fen, last_known_fen):
    """
    Match detected FEN with expected next positions
    """
    board = chess.Board(last_known_fen)
    
    # Generate all legal moves from last known position
    legal_fens = []
    for move in board.legal_moves:
        board.push(move)
        legal_fens.append(board.fen())
        board.pop()
    
    # Find closest match
    best_match = None
    min_diff = float('inf')
    
    for legal_fen in legal_fens:
        diff = count_fen_differences(detected_fen, legal_fen)
        if diff < min_diff:
            min_diff = diff
            best_match = legal_fen
    
    return best_match, min_diff


def count_fen_differences(fen1, fen2):
    """
    Count piece differences between two FEN strings
    """
    # Extract just the board position (first part of FEN)
    board1 = fen1.split()[0]
    board2 = fen2.split()[0]
    
    # Expand FEN notation (convert numbers to dots)
    def expand_fen(fen_board):
        expanded = ""
        for char in fen_board:
            if char.isdigit():
                expanded += '.' * int(char)
            else:
                expanded += char
        return expanded
    
    exp1 = expand_fen(board1).replace('/', '')
    exp2 = expand_fen(board2).replace('/', '')
    
    # Count differences
    differences = sum(1 for a, b in zip(exp1, exp2) if a != b)
    return differences


# Example usage
detected_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
last_known_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

best_match, diff_count = match_with_game_state(detected_fen, last_known_fen)
print(f"Best match: {best_match}")
print(f"Differences: {diff_count}")