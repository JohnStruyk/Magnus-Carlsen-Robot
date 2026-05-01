"""Board-diff pattern classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BLACK_ID = 1
WHITE_ID = 2


@dataclass(frozen=True)
class BoardChange:
    """Structured board-state delta grouped by detected piece color."""

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


def build_board_change(one_removals: Any, two_removals: Any, one_additions: Any, two_additions: Any) -> BoardChange:
    """Create a structured board-change object from raw compare arrays."""
    return BoardChange(one_removals, two_removals, one_additions, two_additions)
