"""Selection helpers for LLM-generated grid_makers.

A selection returned by derive_operations may be a bbox [r,c,h,w] (rectangle) or
this cell-list form for a non-rectangular object. Use sel_of(toindices(obj)) to
Move/Flip/Rotate an object by its exact shape instead of its bounding rectangle.
"""


def sel_of(cells):
    """Wrap an iterable of (r, c) cells as a non-rectangular selection."""
    uniq = sorted({(int(r), int(c)) for r, c in cells})
    return {"cells": [[r, c] for r, c in uniq]}
