"""
ARC Task: 8731374e (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

import numpy as np


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _components(I, min_size):
    """4-connected same-color components, as (color, r0, c0, r1, c1) bboxes."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = I[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(cells) >= min_size:
                ys = [p[0] for p in cells]
                xs = [p[1] for p in cells]
                out.append((int(col), min(ys), min(xs), max(ys), max(xs)))
    return out


def _trim(I, bg, r0, c0, r1, c1):
    """Shave the dirtiest border line (highest fraction of non-bg) until all four are clean."""
    for _ in range(400):
        if r1 - r0 < 4 or c1 - c0 < 4:
            break
        lw = c1 - c0 + 1
        lh = r1 - r0 + 1
        cnt = [
            int(np.count_nonzero(I[r0, c0:c1 + 1] != bg)),
            int(np.count_nonzero(I[r1, c0:c1 + 1] != bg)),
            int(np.count_nonzero(I[r0:r1 + 1, c0] != bg)),
            int(np.count_nonzero(I[r0:r1 + 1, c1] != bg)),
        ]
        frac = [cnt[0] / lw, cnt[1] / lw, cnt[2] / lh, cnt[3] / lh]
        k = max(range(4), key=lambda i: frac[i])
        if cnt[k] == 0:
            break
        if k == 0:
            r0 += 1
        elif k == 1:
            r1 -= 1
        elif k == 2:
            c0 += 1
        else:
            c1 -= 1
    return r0, c0, r1, c1


def _expand(I, bg, r0, c0, r1, c1):
    """Grow outward while the adjacent line still looks like a panel line (<=1 non-bg)."""
    h, w = I.shape
    changed = True
    while changed:
        changed = False
        if r0 > 0 and np.count_nonzero(I[r0 - 1, c0:c1 + 1] != bg) <= 1:
            r0 -= 1
            changed = True
        if r1 < h - 1 and np.count_nonzero(I[r1 + 1, c0:c1 + 1] != bg) <= 1:
            r1 += 1
            changed = True
        if c0 > 0 and np.count_nonzero(I[r0:r1 + 1, c0 - 1] != bg) <= 1:
            c0 -= 1
            changed = True
        if c1 < w - 1 and np.count_nonzero(I[r0:r1 + 1, c1 + 1] != bg) <= 1:
            c1 += 1
            changed = True
    return r0, c0, r1, c1


def _retract(I, bg, r0, c0, r1, c1):
    """Pull sides in until the outer ring is pure bg (the clean frame of the panel)."""
    while r1 - r0 >= 4 and c1 - c0 >= 4:
        if np.any(I[r0, c0:c1 + 1] != bg):
            r0 += 1
            continue
        if np.any(I[r1, c0:c1 + 1] != bg):
            r1 -= 1
            continue
        if np.any(I[r0:r1 + 1, c0] != bg):
            c0 += 1
            continue
        if np.any(I[r0:r1 + 1, c1] != bg):
            c1 -= 1
            continue
        break
    return r0, c0, r1, c1


def _validate(I, bg, r0, c0, r1, c1):
    sub = I[r0:r1 + 1, c0:c1 + 1]
    hh, ww = sub.shape
    if hh < 5 or ww < 5:
        return None
    if (np.any(sub[0] != bg) or np.any(sub[-1] != bg)
            or np.any(sub[:, 0] != bg) or np.any(sub[:, -1] != bg)):
        return None
    mask = sub != bg
    if int(mask.sum()) < 1:
        return None
    if mask.sum(axis=1).max() > 1 or mask.sum(axis=0).max() > 1:
        return None
    fgs = set(int(v) for v in sub[mask].tolist())
    if len(fgs) != 1:
        return None
    return fgs.pop()


def _detect_region(I):
    """Locate the framed panel: a rectangle of one colour whose border ring is clean and
    whose interior holds at most one speck per row and per column (that is exactly what
    the generator builds, and what the verifier's rotate-and-peel loop converges to)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    seeds = []
    for col, r0, c0, r1, c1 in _components(I, 16):
        seeds.append((col, r0, c0, r1, c1))
    for col in sorted(set(int(v) for v in I.flatten().tolist())):
        if np.count_nonzero(I == col) >= 20:
            seeds.append((col, 0, 0, h - 1, w - 1))
    best = None
    tried = set()
    for bg, a0, b0, a1, b1 in seeds:
        key = (bg, a0, b0, a1, b1)
        if key in tried:
            continue
        tried.add(key)
        r0, c0, r1, c1 = _trim(I, bg, a0, b0, a1, b1)
        r0, c0, r1, c1 = _expand(I, bg, r0, c0, r1, c1)
        r0, c0, r1, c1 = _retract(I, bg, r0, c0, r1, c1)
        fg = _validate(I, bg, r0, c0, r1, c1)
        if fg is None:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area > best[0]:
            best = (area, r0, c0, r1, c1, bg, fg)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5], best[6]


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # bgc = panel colour, fgc = speck colour (the rule is stated in these two),
    # ccols = the noise palette; all fixed once per episode.
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    ccols = cols[:]
    random.shuffle(ccols)
    return {"bgc": bgc, "fgc": fgc, "ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, ccols=None) -> dict:
    if ccols is None:
        ccols = list(range(10))
        random.shuffle(ccols)
    max_h = max(10, min(30, int(max_h)))
    max_w = max(10, min(30, int(max_w)))

    last = None
    for _attempt in range(400):
        h = _unifint(diff_lb, diff_ub, (10, max_h))
        w = _unifint(diff_lb, diff_ub, (10, max_w))
        inh = random.randint(5, h - 2)
        inw = random.randint(5, w - 2)
        num = _unifint(diff_lb, diff_ub, (1, min(inh, inw)))

        # diagonal of specks, rows shuffled, transposed, rows shuffled, transposed back
        # -> at most one speck per row and per column
        mh, mw = inh - 2, inw - 2
        mat = [[bgc] * mw for _ in range(mh)]
        for i in range(num):
            if i < mh and i < mw:
                mat[i][i] = fgc
        random.shuffle(mat)
        mat = [list(r) for r in zip(*mat)]      # dmirror
        random.shuffle(mat)
        mat = [list(r) for r in zip(*mat)]      # dmirror back

        sgi = [[bgc] * inw for _ in range(inh)]
        for i in range(mh):
            for j in range(mw):
                sgi[i + 1][j + 1] = mat[i][j]

        dots = [(i, j) for i in range(inh) for j in range(inw) if sgi[i][j] != bgc]
        go = [row[:] for row in sgi]
        for (i, j) in dots:
            for jj in range(inw):
                go[i][jj] = fgc
            for ii in range(inh):
                go[ii][j] = fgc

        numci = _unifint(diff_lb, diff_ub, (3, 10))
        numc = 13 - numci
        cc = ccols[:numc]
        remcols = [x for x in cc if x != bgc]
        if not remcols:
            continue

        gi = [[random.choice(cc) for _ in range(w)] for _ in range(h)]
        loci = random.randint(1, h - inh - 1)
        locj = random.randint(1, w - inw - 1)
        for i in range(inh):
            for j in range(inw):
                gi[loci + i][locj + j] = sgi[i][j]
        a, b = loci, locj
        c, d = loci + inh - 1, locj + inw - 1
        # one non-panel cell hugging each side, so the panel's edges are unambiguous
        for p in ((a - 1, random.randint(b, d)), (random.randint(a, c), b - 1),
                  (c + 1, random.randint(b, d)), (random.randint(a, c), d + 1)):
            gi[p[0]][p[1]] = random.choice(remcols)

        last = {"input": gi, "output": go}
        det = _detect_region(np.array(gi, dtype=int))
        if det is None:
            continue
        if (det[0], det[1], det[2], det[3], det[4]) != (a, b, c, d, bgc):
            continue        # noise made the panel edge ambiguous -> resample
        return {"input": gi, "output": go}

    return last


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    r0, c0, r1, c1, bgc, fgc = _detect_region(I)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    panel = I[r0:r1 + 1, c0:c1 + 1]
    dots = [(int(r), int(c)) for r in range(h) for c in range(w) if panel[r, c] != bgc]
    rows = sorted(set(r for r, _ in dots))
    cols = sorted(set(c for _, c in dots))

    ops, sels = [], []

    # 1. keep only the framed panel (bbox == exactly that full rectangle)
    ops.append(33)
    sels.append([r0, c0, h - 1, w - 1])

    # 2. each speck sends a frontier along its row: paint those rows whole
    for r in rows:
        ops.append(fgc)
        sels.append([r, 0, 0, w - 1])              # exactly the full row r

    cur = panel.copy()
    for r in rows:
        cur[r, :] = fgc

    # 3. quarter-turn clockwise: the panel's columns now stand as rows, so the
    #    remaining frontiers get drawn by exactly the same gesture as the first ones
    rot_state = np.rot90(cur, 3)
    use_rotation = not (h == w and np.array_equal(rot_state, cur))
    sq = max(h, w)
    if use_rotation:
        if h != w:
            ops.append(33)
            sels.append([0, 0, sq - 1, sq - 1])     # pad canvas to a square to turn it
        ops.append(25)                              # Rotate CW, whole square selection
        sels.append([0, 0, sq - 1, sq - 1])
        if h != w:
            ops.append(33)
            sels.append([0, sq - h, w - 1, h - 1])  # crop back to the turned panel
        for c in cols:
            ops.append(fgc)
            sels.append([c, 0, 0, h - 1])           # full row c == the panel's column c
        for c in cols:
            rot_state[c, :] = fgc
        # 4. quarter-turn back, so the panel stands as it did
        back = np.rot90(rot_state, 1)
        if h != w or not np.array_equal(back, rot_state):
            if h != w:
                ops.append(33)
                sels.append([0, 0, sq - 1, sq - 1])     # pad to a square again
            ops.append(24)                              # Rotate CCW, whole square
            sels.append([0, 0, sq - 1, sq - 1])
            if h != w:
                ops.append(33)
                sels.append([sq - h, 0, h - 1, w - 1])  # crop back to the panel
    else:
        # turning would leave the picture identical -> no-op; draw the columns in place
        for c in cols:
            ops.append(fgc)
            sels.append([0, c, h - 1, 0])           # exactly the full column c

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8731374e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8731374e"
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
                                f"for task 8731374e"
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
                    f"Failed to build a complete episode for task 8731374e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8731374e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
