import random
import time

import chess

from pickup_board_piece import move_piece, capture_piece, stage_from_graveyard
from utils.zed_camera import ZedCamera


def execute_robot_move_on_board(board: chess.Board, move: chess.Move, zed: ZedCamera) -> None:
    """Execute one legal move physically using existing pickup logic."""
    move_string = move.uci()
    print(f"[verify] Executing random move: {move_string}")

    if board.is_castling(move):
        king_from_sq = move.from_square
        king_to_sq = move.to_square
        king_piece = board.piece_at(king_from_sq)
        if king_piece is None:
            raise RuntimeError("Castling failed: king piece not found.")

        if board.is_kingside_castling(move):
            rook_from_sq = chess.H1 if board.turn == chess.WHITE else chess.H8
            rook_to_sq = chess.F1 if board.turn == chess.WHITE else chess.F8
        else:
            rook_from_sq = chess.A1 if board.turn == chess.WHITE else chess.A8
            rook_to_sq = chess.D1 if board.turn == chess.WHITE else chess.D8

        rook_piece = board.piece_at(rook_from_sq)
        if rook_piece is None:
            raise RuntimeError("Castling failed: rook piece not found.")

        move_piece(
            king_piece.symbol(),
            chess.square_name(king_from_sq),
            chess.square_name(king_to_sq),
            zed,
        )
        stage_from_graveyard(rook_piece.symbol(), zed)
        move_piece(
            rook_piece.symbol(),
            chess.square_name(rook_from_sq),
            chess.square_name(rook_to_sq),
            zed,
        )
        return

    from_sq = move.from_square
    to_sq = move.to_square
    from_piece = board.piece_at(from_sq)
    to_piece = board.piece_at(to_sq)

    if from_piece is None:
        raise RuntimeError(f"No piece on from-square {chess.square_name(from_sq)}")

    from_symbol = from_piece.symbol()
    from_square = chess.square_name(from_sq)
    to_square = chess.square_name(to_sq)

    if to_piece is not None:
        capture_count = sum(1 for p in board.piece_map().values() if p.color == chess.WHITE)
        capture_count = 16 - capture_count
        capture_piece(from_symbol, to_piece.symbol(), from_square, to_square, zed, capture_count)
    else:
        move_piece(from_symbol, from_square, to_square, zed)


def main() -> None:
    zed = ZedCamera()
    board = chess.Board()

    try:
        while True:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                print("[verify] No legal moves left. Resetting board to start position.")
                board = chess.Board()
                time.sleep(1.0)
                continue

            move = random.choice(legal_moves)
            execute_robot_move_on_board(board, move, zed)
            board.push(move)
            print(f"[verify] Board FEN after move: {board.fen()}")
            time.sleep(0.8)
    finally:
        zed.close()


if __name__ == "__main__":
    main()
