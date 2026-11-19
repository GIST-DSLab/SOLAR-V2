"""
ARC Task: 2bcee788 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    if lo < a:
        lo = a
    return random.randint(lo, hi)


DIRECTIONS = ["right", "left", "down", "up"]   # side of the object the marker line sits on


# ----------------------------------------------------------------------------
# 1. colors  (+ per-instance structural plan: which side the marker line is on)
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]          # 3 is reserved for the output background
    bgc, sepc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRECTIONS):
        examples = [{"direction": d} for d in DIRECTIONS]
        examples += [{"direction": random.choice(DIRECTIONS)}
                     for _ in range(n_ex - len(DIRECTIONS))]
        random.shuffle(examples)
    else:
        examples = [{"direction": d} for d in random.sample(DIRECTIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sepc": sepc, "objc": objc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, sepc, objc, direction=None) -> dict:
    if direction is None:
        direction = random.choice(DIRECTIONS)
    transposed = direction in ("down", "up")

    # limits for the PRE-transform canvas (rows, cols)
    lim_h = max_w if transposed else max_h
    lim_w = max_h if transposed else max_w

    h_hi = max(2, min(20, lim_h - 1))
    w_hi = max(2, min(10, (lim_w - 1) // 2))
    h = _unifint(diff_lb, diff_ub, (2, h_hi))
    w = _unifint(diff_lb, diff_ub, (2, w_hi))

    fullh = _unifint(diff_lb, diff_ub, (h + 1, max(h + 1, lim_h)))
    fullw = _unifint(diff_lb, diff_ub, (2 * w + 1, max(2 * w + 1, lim_w)))

    # ---- build the blob, anchored on the right border column of the mini grid
    spi = random.randrange(h)
    sp = (spi, w - 1)
    shp = {sp}
    rem = {(i, j) for i in range(h) for j in range(w)} - {sp}

    def nbrs(s):
        out = set()
        for (i, j) in s:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            out.add((ni, nj))
        return out

    numcellsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    for _ in range(numc):
        cand = list((rem - shp) & nbrs(shp))
        if not cand:
            break
        shp.add(random.choice(cand))
    guard = 0
    while len({j for _, j in shp}) == 1 and guard < 200:
        guard += 1
        cand = list((rem - shp) & nbrs(shp))
        if not cand:
            break
        shp.add(random.choice(cand))

    gi = np.full((fullh, fullw), bgc, dtype=int)
    go = np.full((fullh, fullw), bgc, dtype=int)
    for (i, j) in shp:
        gi[i, j] = objc
        go[i, j] = objc
        go[i, 2 * w - 1 - j] = objc          # mirrored copy in the output
    for (i, j) in shp:
        if j == w - 1:
            gi[i, w] = sepc                  # marker line = mirrored border column

    if direction == "left":
        gi = np.fliplr(gi); go = np.fliplr(go)
    elif direction == "down":
        gi = gi.T.copy(); go = go.T.copy()
    elif direction == "up":
        gi = np.flipud(gi.T); go = np.flipud(go.T)

    if random.random() < 0.5:                # extra variation, keeps the marker side
        if transposed:
            gi = np.fliplr(gi); go = np.fliplr(go)
        else:
            gi = np.flipud(gi); go = np.flipud(go)

    go = np.where(go == bgc, 3, go)
    return {"input": np.array(gi).tolist(), "output": np.array(go).tolist()}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- colour roles -------------------------------------------------------
    info = {}
    for c in np.unique(I):
        cells = [(int(r), int(cc)) for r, cc in np.argwhere(I == c)]
        rs = [r for r, _ in cells]
        cs = [cc for _, cc in cells]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        info[int(c)] = (cells, area)

    # background = the colour whose bounding box spans the whole canvas
    bgc = max(info, key=lambda c: (info[c][1], len(info[c][0])))
    others = sorted([c for c in info if c != bgc], key=lambda c: len(info[c][0]))

    # 1) the whole background becomes 3
    ops.append(3)
    sels.append(sel_of(info[bgc][0]))

    if len(others) >= 2:
        sepc = others[0]          # thin marker line (fewest cells)
        objc = others[-1]         # the blob
        obj_cells = info[objc][0]
        sep_cells = info[sepc][0]

        r0 = min(r for r, _ in obj_cells); r1 = max(r for r, _ in obj_cells)
        c0 = min(c for _, c in obj_cells); c1 = max(c for _, c in obj_cells)

        shares_row = bool({r for r, _ in sep_cells} & {r for r, _ in obj_cells})

        if shares_row:
            # marker is a vertical line -> reflect left/right across it
            wobj = c1 - c0 + 1
            dc = wobj if min(c for _, c in sep_cells) > c0 else -wobj
            mapped = [(r, c0 + c1 - c + dc) for (r, c) in obj_cells]
            keys = sorted({c for _, c in mapped}, reverse=(dc < 0))
            groups = [[(r, c) for (r, c) in mapped if c == k] for k in keys]
        else:
            # marker is a horizontal line -> reflect up/down across it
            hobj = r1 - r0 + 1
            dr = hobj if min(r for r, _ in sep_cells) > r0 else -hobj
            mapped = [(r0 + r1 - r + dr, c) for (r, c) in obj_cells]
            keys = sorted({r for r, _ in mapped}, reverse=(dr < 0))
            groups = [[(r, c) for (r, c) in mapped if r == k] for k in keys]

        # 2) draw the mirrored blob, line by line, growing outward from the marker
        for grp in groups:
            cells = [(r, c) for (r, c) in grp if 0 <= r < H and 0 <= c < W]
            if not cells:
                continue
            ops.append(int(objc))
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
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
                                f"for task 2bcee788"
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
                    f"Failed to build a complete episode for task 2bcee788 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bcee788-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
