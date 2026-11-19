"""
ARC Task: 85c4e7cd (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
def sample_colors(num_examples=None) -> dict:
    # The generator draws the palette `cols` at random — fix it once per episode so
    # every instance shares one colour scheme. `colord` (which colour sits on which
    # ring) has to stay per-instance: its length is min(h, w), which changes with
    # each instance's shape. The rule — reverse the ring colour order — is
    # colour-independent, so the episode is still learnable.
    ncols = random.randint(2, 10)
    cols = random.sample(list(range(10)), ncols)
    return {"cols": cols}


def generate(diff_lb, diff_ub, max_h, max_w, cols=None, **kwargs) -> dict:
    if not cols or len(cols) < 2:
        cols = random.sample(list(range(10)), 2)

    hmax = max(2, min(15, max_h // 2))
    wmax = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    n = min(h, w)

    # A palindromic ring sequence would make the output identical to the input
    # (an instance with nothing to do) — resample until reversing changes something.
    colord = [random.choice(cols) for _ in range(n)]
    for _ in range(200):
        if colord != colord[::-1]:
            break
        colord = [random.choice(cols) for _ in range(n)]
    if colord == colord[::-1]:
        colord[0], colord[-1] = cols[0], cols[1]

    H, W = 2 * h, 2 * w
    gi = [[0] * W for _ in range(H)]
    go = [[0] * W for _ in range(H)]
    for idx in range(n):
        ci = colord[idx]
        co = colord[n - 1 - idx]
        r0, c0 = idx, idx
        r1, c1 = H - 1 - idx, W - 1 - idx
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r == r0 or r == r1 or c == c0 or c == c1:
                    gi[r][c] = ci
                    go[r][c] = co
    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    """The board is an onion of nested rectangular rings; the rule reverses the
    order of their colours. Ring idx is the border of the rectangle
    rows idx..H-1-idx / cols idx..W-1-idx, and the rings tile the board (the
    innermost rectangle is only two cells thick), so every cell belongs to exactly
    one ring. Everything below is measured on I — the answer grid is never read."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    n = min(H, W) // 2
    cur = [int(I[i, i]) for i in range(n)]       # colour each ring wears in I
    tgt = [cur[n - 1 - i] for i in range(n)]     # reversed order = what each must wear

    ops, sels = [], []
    now = list(cur)                              # colour of each ring as the grid stands

    # ── which way to draw it ──────────────────────────────────────────────────
    # Two people redrawing this onion honestly differ: one floods the board with
    # the new outer colour and draws the smaller rectangles on top of it, another
    # recolours ring by ring from the rim inwards, another starts at the core.
    # The puzzle prefers none of them, so read the choice off the board itself —
    # its size, its number of rings and the colours of its outermost and innermost
    # ring. The same board always gets the same route, and the answer grid has no
    # say in it.
    route = (H + W + n + cur[0] + cur[-1]) % 3

    # The base-and-draw route only tells the truth while the rings really do tile
    # the board; if a core were left over it would get painted too.
    if route == 0 and min(H, W) - 2 * (n - 1) > 2:
        route = 1

    def paint_ring(idx, from_rim):
        """Recolour ring idx on the grid as it stands now."""
        r0, c0 = idx, idx
        r1, c1 = H - 1 - idx, W - 1 - idx
        col = tgt[idx]
        # One flood fill covers the whole ring exactly when both neighbouring rings
        # currently wear a different colour — otherwise the fill would spill.
        free_out = (idx == 0) or (now[idx - 1] != now[idx])
        free_in = (idx == n - 1) or (now[idx + 1] != now[idx])
        if free_out and free_in:
            # poke the fill at the corner the sweep is coming from
            sr, sc = (r0, c0) if from_rim else (r1, c1)
            ops.append(10 + col); sels.append([sr, sc, 0, 0])
        elif (c1 - c0) >= (r1 - r0):
            # a wide ring reads as two long horizontal bars closed by two short
            # vertical pieces (each selection is exactly that full rectangle)
            ops.append(col); sels.append([r0, c0, 0, c1 - c0])          # top bar
            ops.append(col); sels.append([r1, c0, 0, c1 - c0])          # bottom bar
            if r1 - r0 >= 2:
                ops.append(col); sels.append([r0 + 1, c0, r1 - r0 - 2, 0])   # left side
                ops.append(col); sels.append([r0 + 1, c1, r1 - r0 - 2, 0])   # right side
        else:
            # a tall ring reads the other way round: two long vertical bars first,
            # then the short pieces that close them top and bottom
            ops.append(col); sels.append([r0, c0, r1 - r0, 0])          # left bar
            ops.append(col); sels.append([r0, c1, r1 - r0, 0])          # right bar
            if c1 - c0 >= 2:
                ops.append(col); sels.append([r0, c0 + 1, 0, c1 - c0 - 2])   # top piece
                ops.append(col); sels.append([r1, c0 + 1, 0, c1 - c0 - 2])   # bottom piece
        now[idx] = col

    if route == 0:
        # Lay the new rim colour over the whole board, then draw each next
        # rectangle on top of it, biggest first: what stays visible of each
        # rectangle is exactly one ring.
        for idx in range(n):
            if all(now[j] == tgt[idx] for j in range(idx, n)):
                continue                       # that rectangle already wears this colour
            # selection is exactly the full rectangle rows idx..H-1-idx
            ops.append(tgt[idx])
            sels.append([idx, idx, H - 1 - 2 * idx, W - 1 - 2 * idx])
            for j in range(idx, n):
                now[j] = tgt[idx]
    elif route == 1:
        # ring by ring, starting at the rim and working inwards
        for idx in range(n):
            if now[idx] != tgt[idx]:
                paint_ring(idx, True)
    else:
        # ring by ring, starting at the core and working outwards
        for idx in range(n - 1, -1, -1):
            if now[idx] != tgt[idx]:
                paint_ring(idx, False)

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task 85c4e7cd"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task 85c4e7cd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"85c4e7cd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
