import chess
import chess.engine
import chess.svg
import webbrowser
import os

def get_best_move(fen_string, time_limit=2.0):
    # 1. Path to your downloaded Stockfish executable
    # Update this to your actual path (e.g., "./stockfish/stockfish-windows-x86-64.exe")
    engine_path = r"/home/rob/Desktop/stockfish/stockfish-ubuntu-x86-64-avx2"
    # 2. Initialize the engine
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        
        # 3. Create a board object from your FEN
        board = chess.Board(fen_string)
        
        # 4. Ask Stockfish for the best move
        # 'limit' controls how long it thinks (longer = more accurate)
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        
        return result.move

def visualize_board(board, best_move=None):
    # Generate the SVG data
    # 'arrows' draws a blue arrow from the start square to the end square
    arrows = []
    if best_move:
        arrows = [chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#0000cccc")]
    
    board_svg = chess.svg.board(
        board=board,
        arrows=arrows,
        size=400  # Size in pixels
    )
    
    # Save to a temporary file
    output_path = "current_board.svg"
    with open(output_path, "w") as f:
        f.write(board_svg)
    
    # Open the file in your default web browser automatically
    webbrowser.open("file://" + os.path.realpath(output_path))

# Example Usage:
current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
best_move = get_best_move(current_fen)

print(f"Stockfish recommends: {best_move}")
# Output will be in UCI format (e.g., 'e2e4')

board = chess.Board(current_fen)
visualize_board(board, best_move)