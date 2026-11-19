"""
ARC Task: 025d127b (RE-ARC) — LLM-generated grid_maker
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
import random
import numpy as np
from collections import Counter, deque
from maker.sel_helpers import sel_of


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    v = a + int(random.uniform(lb, ub) * (b - a + 1))
    return max(a, min(b, v))


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    if c0 == c1:
        return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
    if abs(r1 - r0) == abs(c1 - c0):
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        return {(r0 + i * sr, c0 + i * sc) for i in range(abs(r1 - r0) + 1)}
    return set()


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccols = random.sample(rem, random.randint(1, 9))
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols=None, **kw) -> dict:
    if not ccols:
        ccols = [c for c in range(10) if c != bgc][:1]
    h = _unifint(diff_lb, diff_ub, (5, max_h))
    w = _unifint(diff_lb, diff_ub, (5, max_w))
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    maxtr = 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        lr, lc = random.choice(sorted(cands))
        topl = _connect((0, 0), (0, ow - 1))
        leftl = _connect((1, 0), (oh - 2, oh - 3))
        rightl = _connect((1, ow), (oh - 2, ow + oh - 3))
        botl = _connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        rm = max(c for _, c in inobj)
        outobj = ({(r, c + 1) for r, c in topl} | botl | {(r, c + 1) for r, c in leftl}
                  | _connect((1, ow + 1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)})
        outobj = {ij for ij in outobj if ij[1] <= rm}
        full = {(r + lr, c + lc) for r, c in inobj | outobj}
        inobj = {(r + lr, c + lc) for r, c in inobj}
        outobj = {(r + lr, c + lc) for r, c in outobj}
        if full <= inds:
            nb = set()
            for r, c in full:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb.add((r + dr, c + dc))
            inds = inds - full - nb
            succ += 1
            col = random.choice(ccols)
            for r, c in inobj:
                gi[r][c] = col
            for r, c in outobj:
                go[r][c] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour (objects are sparse outlines, so majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def comps(conn8):
        seen = np.zeros((hi, wi), bool)
        out = []
        nbr = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)] \
            if conn8 else [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(hi):
            for c in range(wi):
                if seen[r, c] or I[r, c] == bgc:
                    continue
                col = I[r, c]
                seen[r, c] = True
                q = deque([(r, c)])
                cells = []
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for dr, dc in nbr:
                        x, y = a + dr, b + dc
                        if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True
                            q.append((x, y))
                out.append((int(col), cells))
        return out

    objs = comps(True)       # whole outlines (same colour, 8-connected)
    pieces = comps(False)    # straight 4-connected strokes
    piece_of = {}
    for k, (_, cells) in enumerate(pieces):
        for cell in cells:
            piece_of[cell] = k

    movers_nz, movers_z = [], []
    for col, cells in objs:
        ks = sorted({piece_of[cell] for cell in cells})
        # the stroke reaching furthest right (bottom edge + corner) is the anchor
        keeper = max(ks, key=lambda k: max(c for _, c in pieces[k][1]))
        mv = [cell for k in ks if k != keeper for cell in pieces[k][1]]
        (movers_z if col == 0 else movers_nz).extend(mv)

    ops, sels = [], []
    holes = []
    if movers_nz:
        # one MoveR carries every non-anchor stroke of every outline at once
        ops.append(22)
        sels.append(sel_of(sorted(movers_nz)))
        s = set(movers_nz)
        holes += [cell for cell in movers_nz if (cell[0], cell[1] - 1) not in s]
    if movers_z:
        # colour 0 is invisible to ARCLE's object buffer: draw the shifted strokes
        s = set(movers_z)
        landing = sorted({(r, c + 1) for r, c in movers_z} - s)
        if landing:
            ops.append(0)
            sels.append(sel_of(landing))
        holes += [cell for cell in movers_z if (cell[0], cell[1] - 1) not in s]
    if bgc != 0 and holes:
        # the vacated left edge of every moved stroke goes back to background
        ops.append(bgc)
        sels.append(sel_of(sorted(holes)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 025d127b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 025d127b"
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
                                f"for task 025d127b"
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
                    f"Failed to build a complete episode for task 025d127b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"025d127b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
