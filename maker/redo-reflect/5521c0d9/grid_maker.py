"""
ARC Task: 5521c0d9 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # fallback: mask expressed as explicit cell list
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# Discrete structural variant: the whole scene is rotated by one of 4 rotations,
# which decides WHICH edge the bars are anchored to (bottom / left / top / right).
VARIANTS = [{"rot": "identity"}, {"rot": "rot90"}, {"rot": "rot180"}, {"rot": "rot270"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(2, 9)
    ccols = random.sample(remcols, ncols)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was shown
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, rot=None, **kw) -> dict:
    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def rot_cw(g):
        H = len(g); W = len(g[0])
        return [[g[H - 1 - j][i] for j in range(H)] for i in range(W)]

    def rot_ccw(g):
        H = len(g); W = len(g[0])
        return [[g[j][W - 1 - i] for j in range(H)] for i in range(W)]

    def rot_180(g):
        return [row[::-1] for row in g[::-1]]

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(2, 9))
    ccols = list(ccols)
    if rot is None:
        rot = random.choice([v["rot"] for v in VARIANTS])

    max_h = min(30, int(max_h))
    max_w = min(30, int(max_w))

    # rot90/rot270 swap the final grid dimensions
    if rot in ("rot90", "rot270"):
        cap_h, cap_w = max_w, max_h
    else:
        cap_h, cap_w = max_h, max_w
    if cap_h < 4 or cap_w < 6:
        rot = "identity"
        cap_h, cap_w = max_h, max_w
    cap_h = max(4, cap_h)
    cap_w = max(6, cap_w)

    h = _unifint(diff_lb, diff_ub, (4, cap_h))
    w = _unifint(diff_lb, diff_ub, (6, cap_w))

    inds = list(range(w))
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
    speps = random.sample(inds, nobjs * 2)
    guard = 0
    while (0 in speps) or ((w - 1) in speps):
        guard += 1
        if guard > 2000:
            nobjs = 1
            speps = sorted(random.sample(inds[1:-1], 2))
            break
        nobjs = _unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
        speps = random.sample(inds, nobjs * 2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    forb = -1
    for sp, ep in zip(starts, ends):
        cands = [c for c in ccols if c != forb]
        if not cands:
            cands = list(ccols)
        col = random.choice(cands)
        forb = col
        hdev = _unifint(diff_lb, diff_ub, (0, h // 2))
        hei = random.choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        for r in range(h - hei, h):
            for c in range(sp, ep + 1):
                gi[r][c] = col
        for r in range(h - 2 * hei, h - hei):   # reflected across the bar's far edge, clipped
            if 0 <= r < h:
                for c in range(sp, ep + 1):
                    go[r][c] = col

    if rot == "rot90":
        gi, go = rot_cw(gi), rot_cw(go)
    elif rot == "rot180":
        gi, go = rot_180(gi), rot_180(go)
    elif rot == "rot270":
        gi, go = rot_ccw(gi), rot_ccw(go)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Rule: every bar grows out of one grid edge; it is REFLECTED across its own
    far edge, i.e. the strip made of the bar plus an equally long stretch of
    background beyond it is mirrored.  That mirror is one Flip on that strip.
    When the mirror image would run off the grid, the strip is clipped to the
    grid and the part of the bar whose reflection landed off-grid is erased."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    # bars never touch the two edges parallel to them, so all 4 corners are background
    bgc = int(I[0, 0])

    # --- find the bars (solid one-colour rectangles) -------------------------
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            comps.append((min(rs), min(cs), max(rs), max(cs)))

    ops, sels = [], []
    if not comps:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # --- which edge do all bars grow from? (unique: bars avoid the opposite edge)
    anchor = None
    for cand, test in (("bottom", lambda b: b[2] == h - 1),
                       ("top",    lambda b: b[0] == 0),
                       ("left",   lambda b: b[1] == 0),
                       ("right",  lambda b: b[3] == w - 1)):
        if all(test(b) for b in comps):
            anchor = cand
            break
    if anchor is None:
        anchor = "bottom"

    if anchor in ("bottom", "top"):
        comps.sort(key=lambda b: b[1])      # along the anchoring edge
    else:
        comps.sort(key=lambda b: b[0])

    def rect(r0, c0, r1, c1):
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    for (r0, c0, r1, c1) in comps:
        if anchor == "bottom":
            L = h - r0                       # bar length
            R0 = max(0, h - 2 * L)           # strip = bar + equal stretch above, clipped
            # bbox selection is intentional: the WHOLE strip (bar + background) is mirrored
            ops.append(27); sels.append(sel_of(rect(R0, c0, h - 1, c1)))
            if 2 * L > h:                    # reflection ran off the top edge
                ops.append(bgc); sels.append(sel_of(rect(h - L, c0, L - 1, c1)))
        elif anchor == "top":
            L = r1 + 1
            R1 = min(h - 1, 2 * L - 1)
            ops.append(27); sels.append(sel_of(rect(0, c0, R1, c1)))
            if 2 * L > h:
                ops.append(bgc); sels.append(sel_of(rect(h - L, c0, L - 1, c1)))
        elif anchor == "left":
            L = c1 + 1
            C1 = min(w - 1, 2 * L - 1)
            ops.append(26); sels.append(sel_of(rect(r0, 0, r1, C1)))
            if 2 * L > w:
                ops.append(bgc); sels.append(sel_of(rect(r0, w - L, r1, L - 1)))
        else:  # right
            L = w - c0
            C0 = max(0, w - 2 * L)
            ops.append(26); sels.append(sel_of(rect(r0, C0, r1, w - 1)))
            if 2 * L > w:
                ops.append(bgc); sels.append(sel_of(rect(r0, w - L, r1, L - 1)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
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
                                f"for task 5521c0d9"
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
                    f"Failed to build a complete episode for task 5521c0d9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5521c0d9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
