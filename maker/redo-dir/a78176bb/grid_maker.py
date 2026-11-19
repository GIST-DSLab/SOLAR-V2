"""
ARC Task: a78176bb (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover - fallback
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ---------------------------------------------------------------------------
# 1. episode-level colors + discrete structural plan
#    The only discrete structural variant of this task is the final hmirror:
#    lines run "\" (identity) or "/" (mirrored).  Both must appear in the
#    examples so the test instance is always learnable.
# ---------------------------------------------------------------------------
VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------------------
# 2. generator
# ---------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    hmax = max(6, min(30, int(max_h)))
    wmax = max(6, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (6, hmax))
    w = unifint(diff_lb, diff_ub, (6, wmax))
    nlns = unifint(diff_lb, diff_ub, (1, max(1, (h + w) // 8)))
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    spopts = []
    for idx in range(h - 5, -1, -1):
        spopts.append((idx, 0))
    for idx in range(1, w - 4, 1):
        spopts.append((0, idx))
    lncol = choice(remcols)
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(spopts) == 0:
            break
        sp = choice(spopts)
        ln = shoot(sp, (1, 1)) & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln, key=lambda x: x[0])
        trid1 = randint(2, min(5, len(lno) - 3))
        trid2 = randint(2, min(5, len(lno) - 3))
        tri1 = sfilter(asindices(canvas(-1, (trid1, trid1))), lambda ij: ij[1] >= ij[0])
        triloc1 = add(choice(lno[1:-trid1 - 1]), (0, 1))
        tri2 = dmirror(sfilter(asindices(canvas(-1, (trid2, trid2))), lambda ij: ij[1] >= ij[0]))
        triloc2 = add(choice(lno[1:-trid2 - 1]), (1, 0))
        spo2 = add(sp, (0, -trid2 - 2))
        nexlin2 = {add(spo2, (k, k)) for k in range(max(h, w))} & fullinds
        spo1 = add(sp, (-trid1 - 2, 0))
        nexlin1 = {add(spo1, (k, k)) for k in range(max(h, w))} & fullinds
        inds0 = inds
        for idx, (tri, triloc, nexlin) in enumerate(sample([
            (tri1, triloc1, nexlin1),
            (tri2, triloc2, nexlin2)
        ], 2)):
            tri = shift(tri, triloc)
            fullobj = ln | tri | nexlin
            if idx == 0:
                lncol, tricol = sample(remcols, 2)
            else:
                tricol = choice(remove(lncol, remcols))
            # both variants are checked against the free cells of this round, so
            # nothing ever overlaps / touches another placed structure
            if fullobj.issubset(inds0):
                succ += 1
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                gi = fill(gi, tricol, tri)
                gi = fill(gi, lncol, ln)
                go = fill(go, lncol, ln)
                go = fill(go, lncol, nexlin)
    if mirror is None:
        mirror = choice((True, False))
    if mirror:
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------------------
# 3. ARCLE trajectory
#
# Rule: every right triangle sitting against a diagonal line is an arrow.  It is
# wiped out and replaced by a new full diagonal line, parallel to the line it
# leans on, drawn on the triangle's own side at distance (triangle size + 2).
# The task is stated in the "\" orientation; when the grid arrives mirrored we
# FlipV (hmirror) the whole grid to that orientation, perform the rule there,
# and FlipV back — exactly the reflection the rule is defined through.
# ---------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []
    submit_sel = [0, 0, ho - 1, wo - 1]
    full_rect = [0, 0, h - 1, w - 1]          # whole-grid rectangle (bg included)

    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    def components(g, diagonal):
        if diagonal:
            nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        seen = [[False] * w for _ in range(h)]
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r][c] or int(g[r, c]) == bgc:
                    continue
                col = int(g[r, c])
                seen[r][c] = True
                stack = [(r, c)]
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in nb:
                        a, b = x + dx, y + dy
                        if 0 <= a < h and 0 <= b < w and not seen[a][b] and int(g[a, b]) == col:
                            seen[a][b] = True
                            stack.append((a, b))
                out.append((col, cells))
        return out

    def is_backslash(cells):
        if len(cells) < 2:
            return False
        if len(set(c - r for r, c in cells)) != 1:
            return False
        rs = sorted(r for r, c in cells)
        return rs[-1] - rs[0] == len(cells) - 1

    blobs = components(I, True)
    if not blobs:                                   # empty canvas, nothing to do
        ops.append(34); sels.append(submit_sel)
        return ops, sels

    # orientation: a "\" diagonal line present -> already canonical
    mirrored = not any(is_backslash(cells) for _, cells in blobs)
    G = np.flipud(I).copy() if mirrored else I.copy()

    # triangles = orthogonally connected non-background blobs (line cells are
    # only diagonally adjacent, so they never form such a blob)
    tris = [(col, cells) for col, cells in components(G, False) if len(cells) >= 2]
    if not tris:
        ops.append(34); sels.append(submit_sel)
        return ops, sels

    if mirrored:
        ops.append(27); sels.append(full_rect)      # FlipV = hmirror, whole grid

    tris.sort(key=lambda t: (min(r for r, c in t[1]), min(c for r, c in t[1])))

    for _col, cells in tris:
        r0 = min(r for r, c in cells); c0 = min(c for r, c in cells)
        r1 = max(r for r, c in cells); c1 = max(c for r, c in cells)
        cset = set(cells)
        if (r0, c1) in cset:
            # upper-right triangle: leans on the line running under its top-left,
            # its new line starts just beyond the upper-right corner
            lcol = int(G[r0 + 1, c0]) if r0 + 1 < h else bgc
            sr, sc = r0 - 1, c1 + 1
        else:
            # lower-left triangle: line runs right of its top-left corner,
            # new line starts just beyond the lower-left corner
            lcol = int(G[r0, c0 + 1]) if c0 + 1 < w else bgc
            sr, sc = r1 + 1, c0 - 1

        # the arrow itself disappears
        ops.append(bgc); sels.append(sel_of(sorted(cells)))
        for (r, c) in cells:
            G[r, c] = bgc

        # ... and the parallel diagonal it points at is drawn
        d = sc - sr
        line = [(r, r + d) for r in range(h) if 0 <= r + d < w]
        if line and lcol != bgc:
            ops.append(lcol); sels.append(sel_of(line))
            for (r, c) in line:
                G[r, c] = lcol

    if mirrored:
        ops.append(27); sels.append(full_rect)      # back to the given orientation

    ops.append(34); sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task a78176bb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a78176bb"
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
                                f"for task a78176bb"
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
                    f"Failed to build a complete episode for task a78176bb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a78176bb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
