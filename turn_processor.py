"""Vision grid diffs -> move strings -> python-chess updates (plus callbacks into the arm)."""

from dataclasses import dataclass
from typing import Callable, Sequence

import chess
import numpy as np

BLACK_ID = 1
WHITE_ID = 2


@dataclass(frozen=True)
class BoardChange:
    """Counts from compare_board_states: one_* = black markers, two_* = white."""

    one_removals: object
    two_removals: object
    one_additions: object
    two_additions: object

    @property
    def changed(self) -> bool:
        return any(
            len(x) > 0
            for x in (self.one_removals, self.two_removals, self.one_additions, self.two_additions)
        )

    @property
    def is_single_move(self) -> bool:
        return (
            (len(self.one_removals) == 1 and len(self.one_additions) == 1 and len(self.two_removals) == 0 and len(self.two_additions) == 0)
            or (len(self.two_removals) == 1 and len(self.two_additions) == 1 and len(self.one_removals) == 0 and len(self.one_additions) == 0)
        )

    @property
    def is_legal_capture(self) -> bool:
        return (
            (len(self.one_removals) == 1 and len(self.one_additions) == 1 and len(self.two_removals) == 1 and len(self.two_additions) == 0)
            or (len(self.two_removals) == 1 and len(self.two_additions) == 1 and len(self.one_removals) == 1 and len(self.one_additions) == 0)
        )

    @property
    def is_castle(self) -> bool:
        return (
            (len(self.one_removals) == 2 and len(self.one_additions) == 2 and len(self.two_removals) == 0 and len(self.two_additions) == 0)
            or (len(self.two_removals) == 2 and len(self.two_additions) == 2 and len(self.one_removals) == 0 and len(self.one_additions) == 0)
        )

    @property
    def moving_color_val(self) -> int | None:
        if (len(self.one_removals) > 0 or len(self.one_additions) > 0) and len(self.two_removals) == 0 and len(self.two_additions) == 0:
            return BLACK_ID
        if (len(self.two_removals) > 0 or len(self.two_additions) > 0) and len(self.one_removals) == 0 and len(self.one_additions) == 0:
            return WHITE_ID
        return None


def rc_alg(square: Sequence[int]) -> str:
    row, col = int(square[0]), int(square[1])
    return f"{chr(ord('a') + col)}{8 - row}"


def rc_square(rc: Sequence[int]) -> chess.Square:
    r, c = int(rc[0]), int(rc[1])
    return chess.square(c, 7 - r)


def ep_uci(board: chess.Board, mover_rem, mover_add, victim_rem) -> str | None:
    """Requires ``board.turn`` set correctly. Scans legal EP moves only."""
    fs, ts, vs = rc_square(mover_rem), rc_square(mover_add), rc_square(victim_rem)
    for m in board.legal_moves:
        if not board.is_en_passant(m) or m.from_square != fs or m.to_square != ts:
            continue
        # EP victim sits one rank behind the landing square along pawn advance direction.
        cap = m.to_square - 8 if board.turn == chess.WHITE else m.to_square + 8
        if cap == vs:
            return m.uci()
    return None


def castle_from_grid(removals: np.ndarray, additions: np.ndarray) -> str | None:
    # Warp columns where king/rook markers vanished (rcols) and where they appeared (acols).
    if removals.shape[0] != 2 or additions.shape[0] != 2:
        return None
    rcols = sorted(int(x) for x in removals[:, 1])
    acols = sorted(int(x) for x in additions[:, 1])
    if rcols not in ([0, 4], [4, 7]):
        return None
    if acols == [5, 6]:
        return "O-O"
    if acols == [2, 3]:
        return "O-O-O"
    return None


def capture_uci(cap_rem, cap_add, vic_rem) -> str | None:
    if np.array_equal(cap_add, vic_rem):
        return rc_alg(cap_rem) + rc_alg(cap_add)
    return None


def determine_move(
    one_removals,
    two_removals,
    one_additions,
    two_additions,
    board: chess.Board | None = None,
) -> str:
    """Either UCI / O-O / O-O-O, or 'BAD.*' if the grid pattern is nonsense (printed upstream)."""
    if len(one_additions) + len(two_additions) > len(one_removals) + len(two_removals):
        return "BAD. more pieces after than before"
    if len(one_additions) + len(two_additions) + 1 < len(one_removals) + len(two_removals):
        return "BAD. too many removed"
    if len(one_additions) > 0 and len(two_additions) > 0:
        return "BAD. both colors moved"
    if len(one_removals) + len(two_removals) == 0:
        return "BAD. nothing moved"

    if len(one_removals) == 1 and len(one_additions) == 1:
        if len(two_removals) == 1:
            c = capture_uci(one_removals[0], one_additions[0], two_removals[0])
            if c:
                return c
            if board is not None:
                ep = ep_uci(board, one_removals[0], one_additions[0], two_removals[0])
                if ep:
                    return ep
            return "BAD. capture/ep mismatch"
        return rc_alg(one_removals[0]) + rc_alg(one_additions[0])

    if len(one_removals) == 2 and len(one_additions) == 2:
        c = castle_from_grid(one_removals, one_additions)
        return c if c else "BAD. castle pattern"

    if len(two_removals) == 1 and len(two_additions) == 1:
        if len(one_removals) == 1:
            c = capture_uci(two_removals[0], two_additions[0], one_removals[0])
            if c:
                return c
            if board is not None:
                ep = ep_uci(board, two_removals[0], two_additions[0], one_removals[0])
                if ep:
                    return ep
            return "BAD. capture/ep mismatch"
        return rc_alg(two_removals[0]) + rc_alg(two_additions[0])

    if len(two_removals) == 2 and len(two_additions) == 2:
        c = castle_from_grid(two_removals, two_additions)
        return c if c else "BAD. castle pattern"

    return "BAD. unsupported pattern"


def grid_square(row: int, col: int) -> chess.Square:
    # Same row major as detect_pieces: row 0 = rank 8.
    return chess.square(col, 7 - row)


def describe_move(board: chess.Board, removals, additions) -> None:
    for fr, fc in [(tuple(sq), 1) for sq in removals[0]] + [(tuple(sq), 2) for sq in removals[1]]:
        for tr, tc in [(tuple(sq), 1) for sq in additions[0]] + [(tuple(sq), 2) for sq in additions[1]]:
            if fc != tc:
                continue
            fs, ts = grid_square(*fr), grid_square(*tr)
            p = board.piece_at(fs)
            cn = "black" if fc == 1 else "white"
            pn = chess.piece_name(p.piece_type) if p else "unknown"
            print(f"  {cn} {pn} {chess.square_name(fs)} -> {chess.square_name(ts)}")
            return
    print("  (could not pair removal/addition)")


def illegal_reason(board: chess.Board, move: str) -> str:
    """Returns ``legal`` or a short reason string (shared with ``illegal_extra``)."""
    if move.startswith("BAD"):
        return move
    if move in ("O-O", "O-O-O"):
        try:
            m = board.parse_san(move)
        except ValueError:
            return "bad castle"
        return "legal" if m in board.legal_moves else "illegal castle"
    try:
        m = chess.Move.from_uci(move)
    except Exception:
        return f"bad UCI {move!r}"
    if m in board.legal_moves:
        return "legal"
    fp = board.piece_at(m.from_square)
    tp = board.piece_at(m.to_square)
    if fp is None:
        return "empty source"
    if fp.color != board.turn:
        return "wrong color on source"
    if tp is not None and tp.color == board.turn:
        return "blocked by own piece"
    if m not in board.pseudo_legal_moves:
        return "illegal piece motion"
    tb = board.copy(stack=False)
    tb.push(m)
    if tb.is_check():
        return "king in check after move"
    return "special rule (castle/ep/rights)"


def illegal_extra(board: chess.Board, move: str) -> None:
    print(f"  ({illegal_reason(board, move)})")
    if move in ("O-O", "O-O-O") or move.startswith("BAD"):
        return
    try:
        m = chess.Move.from_uci(move)
        tg = sorted({chess.square_name(x.to_square) for x in board.legal_moves if x.from_square == m.from_square})
        if tg:
            print(f"  legal from {chess.square_name(m.from_square)}: {', '.join(tg)}")
    except Exception:
        pass


def return_pieces(board: chess.Board, one_r, two_r) -> None:
    """Console-only: where to put pieces back after we reject a vision move."""
    for rc, cv in [(tuple(r), 1) for r in one_r] + [(tuple(r), 2) for r in two_r]:
        sq = grid_square(*rc)
        p = board.piece_at(sq)
        cn = "white" if cv == 2 else "black"
        pn = chess.piece_name(p.piece_type) if p else "unknown"
        print(f"  return {cn} {pn} to {chess.square_name(sq)}")


def detect_castling_uci(board: chess.Board, removals, additions) -> str | None:
    rs = {grid_square(int(r), int(c)) for r, c in removals}
    ads = {grid_square(int(r), int(c)) for r, c in additions}
    for mv in board.legal_moves:
        if not board.is_castling(mv):
            continue
        if board.is_kingside_castling(mv):
            rf, rt = (chess.H1, chess.F1) if board.turn == chess.WHITE else (chess.H8, chess.F8)
        else:
            rf, rt = (chess.A1, chess.D1) if board.turn == chess.WHITE else (chess.A8, chess.D8)
        if rs == {mv.from_square, rf} and ads == {mv.to_square, rt}:
            return mv.uci()
    return None


def push_move(board: chess.Board, move: str) -> None:
    if move in ("O-O", "O-O-O"):
        board.push_san(move)
    else:
        board.push_uci(move)


@dataclass(frozen=True)
class TurnProcessResult:
    prior_board_state: object
    game_over: bool


def process_detected_change(
    chess_board: chess.Board,
    turn: chess.Color,
    board_state,
    change: BoardChange,
    try_robot_reply: Callable[[str], bool],
    on_game_over: Callable[[chess.Board], None],
) -> TurnProcessResult:
    """The only place vision touches chess_board + optional try_robot_reply()."""
    one_r, two_r, one_a, two_a = (
        change.one_removals,
        change.two_removals,
        change.one_additions,
        change.two_additions,
    )
    mv_color = change.moving_color_val  # which marker color moved in vision (1=black 2=white)
    expect = WHITE_ID if turn == chess.WHITE else BLACK_ID  # python-chess side to move
    if mv_color is not None and mv_color != expect:
        wc = "black" if mv_color == BLACK_ID else "white"
        rc = "white" if turn == chess.WHITE else "black"
        print(f"Wrong turn: {wc} moved, {rc} to play.")
        if mv_color == BLACK_ID and turn == chess.WHITE:
            parts = []
            for rc_ in one_r:
                sq = grid_square(*tuple(rc_))
                p = chess_board.piece_at(sq)
                parts.append(f"{chess.piece_name(p.piece_type) if p else '?'}@{chess.square_name(sq)}")
            if parts:
                print("Black moved on White's turn: " + ", ".join(parts))
        return_pieces(chess_board, one_r, two_r)
        return TurnProcessResult(prior_board_state=board_state, game_over=False)

    if change.is_castle:
        cc = BLACK_ID if len(one_r) == 2 else WHITE_ID  # castle detected on black markers vs white
        rem, add = (one_r, one_a) if cc == BLACK_ID else (two_r, two_a)
        c_uci = detect_castling_uci(chess_board, rem, add)
        if c_uci is None:
            print("Castling pattern but no legal castle. Reset pieces.")
            return TurnProcessResult(prior_board_state=board_state, game_over=False)
        print(f"Castle: {c_uci}")
        try:
            chess_board.push_uci(c_uci)
            turn = chess.BLACK if turn == chess.WHITE else chess.WHITE  # side after castle push
        except Exception as exc:
            print(f"  ILLEGAL CASTLE ({exc})")
            return TurnProcessResult(prior_board_state=board_state, game_over=False)
        if turn == chess.BLACK and try_robot_reply("Black (robot) move failed"):
            return TurnProcessResult(prior_board_state=board_state, game_over=True)
        return TurnProcessResult(prior_board_state=board_state, game_over=False)

    if not change.is_single_move and not change.is_legal_capture:
        if len(one_r) or len(two_r):
            print("Invalid change. Reset pieces.")
            return_pieces(chess_board, one_r, two_r)
        return TurnProcessResult(prior_board_state=board_state, game_over=False)

    move = determine_move(one_r, two_r, one_a, two_a, chess_board)
    print(f"Move: {move}" + (" (White)" if turn == chess.WHITE else " (Black to move)"))
    describe_move(chess_board, (one_r, two_r), (one_a, two_a))

    try:
        rsn = illegal_reason(chess_board, move)
        if rsn != "legal":
            raise ValueError(rsn)
        push_move(chess_board, move)
        turn = chess.BLACK if turn == chess.WHITE else chess.WHITE  # toggled after accepted push
        if chess_board.is_check():
            cs = "White" if chess_board.turn == chess.WHITE else "Black"
            print(f"Check: {cs}")
        if chess_board.is_game_over(claim_draw=True):
            on_game_over(chess_board)
            return TurnProcessResult(prior_board_state=board_state, game_over=True)
    except Exception as exc:
        print(f"  ILLEGAL: {exc} | candidate {move}")
        print(f"  FEN: {chess_board.fen()}")
        illegal_extra(chess_board, move)
        return_pieces(chess_board, one_r, two_r)
        return TurnProcessResult(prior_board_state=board_state, game_over=False)

    if turn == chess.BLACK and try_robot_reply("Black (robot) move failed"):
        return TurnProcessResult(prior_board_state=board_state, game_over=True)

    return TurnProcessResult(prior_board_state=board_state, game_over=False)
