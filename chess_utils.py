import numpy as np

def square_to_alg(square):
    r, c = square
    fyle = chr(ord('a') + c)
    rank = str(8 - r)
    return fyle + rank

def determine_castle(removals, additions):
    if removals[0, 1] == 4 and removals[1, 1] % 7 == 0:
        if additions[0, 1] == 5 and additions[1, 1] == 6:
            return 'O-O'
        if additions[0, 1] == 2 and additions[1, 1] == 3:
            return 'O-O-O'
        
    print("bad csatle")
    return None


def determine_capture(capturing_removal, capturing_addition, captured_removal):
    print("determining capture")
    if np.array_equal(capturing_addition, captured_removal):
        return square_to_alg(capturing_removal) + square_to_alg(capturing_addition)
    print("BAD. Capturing piece didn't go to captured piece's location")
    return None


def determine_normal_move(moving_removal, moving_addition):
    return square_to_alg(moving_removal) + square_to_alg(moving_addition)


def determine_move(one_removals, two_removals, one_additions, two_additions):

    if len(one_additions) + len(two_additions) > len(one_removals) + len(two_removals):
        return "BAD. there are more pieces now than at start of move"

    if len(one_additions) + len(two_additions) + 1 < len(one_removals) + len(two_removals):
        return "BAD. too many pieces removed"

    if len(one_additions) > 0 and len(two_additions) > 0:
        return "BAD. both color pieces have moved."

    if len(one_removals) + len(two_removals) == 0:
        return "BAD. neither color piece has moved."

    if len(one_removals) == 1 and len(one_additions) == 1:
        if len(two_removals) == 1:
            return determine_capture(one_removals[0], one_additions[0], two_removals[0])
        else:
            return determine_normal_move(one_removals[0], one_additions[0])

    if len(one_removals) == 2 and len(one_additions) == 2:
        return determine_castle(one_removals, one_additions)

    if len(two_removals) == 1 and len(two_additions) == 1:
        if len(one_removals) == 1:
            return determine_capture(two_removals[0], two_additions[0], one_removals[0])
        else:
            return determine_normal_move(two_removals[0], two_additions[0])

    if len(two_removals) == 2 and len(two_additions) == 2:
        return determine_castle(two_removals, two_additions)