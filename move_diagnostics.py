"""Move analysis and diagnostics helpers for the game loop."""

from __future__ import annotations

import chess


def row_col_to_chess_square(row: int, col: int) -> chess.Square:
    """Convert board-state ``(row, col)`` into a python-chess square."""
    return chess.square(col, 7 - row)


def describe_move(chess_board: chess.Board, removals, additions) -> None:
    """Print piece/color/source/destination inferred from board-state deltas."""
    all_removals = [(tuple(sq), 1) for sq in removals[0]] + [(tuple(sq), 2) for sq in removals[1]]
    all_additions = [(tuple(sq), 1) for sq in additions[0]] + [(tuple(sq), 2) for sq in additions[1]]

    for from_rc, from_color in all_removals:
        for to_rc, to_color in all_additions:
            if from_color != to_color:
                continue
            from_sq = row_col_to_chess_square(*from_rc)
            to_sq = row_col_to_chess_square(*to_rc)
            piece = chess_board.piece_at(from_sq)
            color_name = "black" if from_color == 1 else "white"
            piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
            print(
                f"  {color_name} {piece_name} moved from "
                f"{chess.square_name(from_sq)} to {chess.square_name(to_sq)}"
            )
            return
    print("  Could not identify moving piece.")


def explain_illegal_move(chess_board: chess.Board, move_uci: str) -> str:
    """Return a human-readable reason why UCI move is illegal."""
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return f"invalid UCI format: '{move_uci}'"

    if move in chess_board.legal_moves:
        return "move is legal"

    from_piece = chess_board.piece_at(move.from_square)
    to_piece = chess_board.piece_at(move.to_square)
    side_to_move = chess_board.turn

    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)

    if from_piece is None:
        return (
            f"Illegal move {move_uci}: source square {from_name} is empty. "
            f"No piece exists on {from_name}, so the move to {to_name} cannot be made."
        )

    if from_piece.color != side_to_move:
        wrong = "white" if from_piece.color == chess.WHITE else "black"
        right = "white" if side_to_move == chess.WHITE else "black"
        return (
            f"Illegal move {move_uci}: turn violation at source {from_name}. "
            f"Piece on {from_name} is {wrong}, but it is {right} to move."
        )

    if to_piece is not None and to_piece.color == side_to_move:
        side = "white" if side_to_move == chess.WHITE else "black"
        return (
            f"Illegal move {move_uci}: destination square {to_name} is blocked by another {side} piece."
        )

    if move not in chess_board.pseudo_legal_moves:
        piece_name = chess.piece_name(from_piece.piece_type)
        return (
            f"Illegal move {move_uci}: piece-rule violation from {from_name} to {to_name}. "
            f"A {piece_name} cannot legally move from {from_name} to {to_name} in this position."
        )

    test_board = chess_board.copy(stack=False)
    test_board.push(move)
    side_name = "white" if side_to_move == chess.WHITE else "black"
    if test_board.is_check():
        king_sq = test_board.king(side_to_move)
        king_name = chess.square_name(king_sq) if king_sq is not None else "unknown"
        attacker_color = not side_to_move
        attackers = list(test_board.attackers(attacker_color, king_sq)) if king_sq is not None else []
        attacker_names = ", ".join(chess.square_name(sq) for sq in attackers) if attackers else "unknown"
        return (
            f"Illegal move {move_uci}: king safety violation after moving {from_name} to {to_name}. "
            f"After {move_uci}, the {side_name} king on {king_name} is in check from: {attacker_names}."
        )

    return (
        f"Illegal move {move_uci}: position-specific rule failure from {from_name} to {to_name}. "
        "Likely castling rights, en passant timing, or another special constraint."
    )


def print_illegal_move_report(chess_board: chess.Board, move_uci: str) -> None:
    """Print detailed move legality diagnostics."""
    print("  ---------------- ILLEGAL MOVE DIAGNOSTIC ----------------")
    print(f"  Position FEN: {chess_board.fen()}")
    side = "white" if chess_board.turn == chess.WHITE else "black"
    print(f"  Side to move: {side}")
    print(f"  Candidate UCI: {move_uci}")

    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        print("  Could not parse UCI string into a chess move.")
        print("  ---------------------------------------------------------")
        return

    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)
    from_piece = chess_board.piece_at(move.from_square)
    to_piece = chess_board.piece_at(move.to_square)
    print(f"  From square: {from_name}")
    print(f"  To square:   {to_name}")
    print(f"  Source piece: {from_piece.symbol() if from_piece else 'empty'}")
    print(f"  Destination occupant: {to_piece.symbol() if to_piece else 'empty'}")
    print(f"  Pseudo-legal: {'yes' if move in chess_board.pseudo_legal_moves else 'no'}")
    print(f"  Legal:        {'yes' if move in chess_board.legal_moves else 'no'}")

    if from_piece is not None:
        legal_targets = sorted(
            {
                chess.square_name(m.to_square)
                for m in chess_board.legal_moves
                if m.from_square == move.from_square
            }
        )
        pseudo_targets = sorted(
            {
                chess.square_name(m.to_square)
                for m in chess_board.pseudo_legal_moves
                if m.from_square == move.from_square
            }
        )
        print(f"  Piece on source: {chess.piece_name(from_piece.piece_type)}")
        print(f"  Legal targets from {from_name}: {', '.join(legal_targets) if legal_targets else 'none'}")
        if pseudo_targets and pseudo_targets != legal_targets:
            print(f"  Pseudo-legal targets from {from_name}: {', '.join(pseudo_targets)}")

    if move in chess_board.pseudo_legal_moves and move not in chess_board.legal_moves:
        test_board = chess_board.copy(stack=False)
        test_board.push(move)
        king_sq = test_board.king(chess_board.turn)
        if king_sq is not None:
            attacker_color = not chess_board.turn
            attackers = sorted(test_board.attackers(attacker_color, king_sq))
            attacker_names = ", ".join(chess.square_name(sq) for sq in attackers) if attackers else "unknown"
            print(f"  King after move: {chess.square_name(king_sq)}")
            print(f"  Checking attacker square(s): {attacker_names}")

    print(f"  Reason: {explain_illegal_move(chess_board, move_uci)}")
    print("  ---------------------------------------------------------")


def print_piece_return_instructions(chess_board: chess.Board, one_removals, two_removals) -> None:
    """Print piece-by-piece return instructions for detected removals."""
    all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
    for rc, color_val in all_removals:
        sq = row_col_to_chess_square(*rc)
        piece = chess_board.piece_at(sq)
        color_name = "white" if color_val == 2 else "black"
        piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
        print(f"  return {color_name} {piece_name} to {chess.square_name(sq)}")


def detect_castling(chess_board: chess.Board, removals, additions) -> str | None:
    """Match a 2-removal/2-addition delta to a legal castling move."""
    removed_squares = {row_col_to_chess_square(r, c) for r, c in [tuple(removals[0]), tuple(removals[1])]}
    added_squares = {row_col_to_chess_square(r, c) for r, c in [tuple(additions[0]), tuple(additions[1])]}
    for move in chess_board.legal_moves:
        if not chess_board.is_castling(move):
            continue
        king_from = move.from_square
        king_to = move.to_square
        if chess_board.is_kingside_castling(move):
            rook_from = chess.H1 if chess_board.turn == chess.WHITE else chess.H8
            rook_to = chess.F1 if chess_board.turn == chess.WHITE else chess.F8
        else:
            rook_from = chess.A1 if chess_board.turn == chess.WHITE else chess.A8
            rook_to = chess.D1 if chess_board.turn == chess.WHITE else chess.D8
        if removed_squares == {king_from, rook_from} and added_squares == {king_to, rook_to}:
            return move.uci()
    return None
