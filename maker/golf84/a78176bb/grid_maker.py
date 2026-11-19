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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None, **kw) -> dict:
    from random import sample, choice, randint
    from dsl import (interval, remove, canvas, asindices, shoot, sfilter, add, dmirror,
                     shift, mapply, neighbors, dneighbors, fill, hmirror)
    from utils import unifint
    if mirror is None:
        mirror = random.choice((True, False))
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    nlns = unifint(diff_lb, diff_ub, (1, (h + w) // 8))
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
            if (
                fullobj.issubset(inds) if idx == 0 else (tri | nexlin).issubset(fullobj)
            ):
                succ += 1
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                # reject ambiguous / colliding layouts (caller retries)
                for (a, b) in (tri | ln):
                    if gi[a][b] != bgc and (gi[a][b] != lncol or (a, b) in tri):
                        raise ValueError("overlapping structures")
                for (a, b) in mapply(dneighbors, tri):
                    if 0 <= a < h and 0 <= b < w and (a, b) not in ln and (a, b) not in tri \
                            and gi[a][b] != bgc:
                        raise ValueError("marker touches another structure")
                for (a, b) in nexlin:
                    if go[a][b] != bgc and go[a][b] != lncol:
                        raise ValueError("ambiguous colliding guide lines")
                gi = fill(gi, tricol, tri)
                gi = fill(gi, lncol, ln)
                go = fill(go, lncol, ln)
                go = fill(go, lncol, nexlin)
    if mirror:
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    A = np.asarray(I, dtype=int)
    h, w = A.shape
    bgc = int(Counter(A.flatten().tolist()).most_common(1)[0][0])

    def components(G, conn8):
        seen = np.zeros((h, w), dtype=bool)
        d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dirs = d4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else d4
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(G[r, c]) == bgc:
                    continue
                col = int(G[r, c])
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append((col, set(cells)))
        return out

    def is_dr_line(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        n = max(rs) - r0
        if n != max(cs) - c0 or len(cells) != n + 1:
            return False
        return cells == {(r0 + k, c0 + k) for k in range(n + 1)}

    def is_straight(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            return cells == {(r0, c) for c in range(c0, c1 + 1)}
        if c0 == c1:
            return cells == {(r, c0) for r in range(r0, r1 + 1)}
        return is_dr_line(cells)

    # Canonical orientation: guide lines run down-right; otherwise the scene is mirrored up-down.
    flip = not any(len(cells) >= 3 and is_dr_line(cells) for _, cells in components(A, True))
    B = A[::-1, :].copy() if flip else A

    def back(rc):
        r, c = rc
        return (h - 1 - r, c) if flip else (r, c)

    # Triangle markers: 4-connected single-colour blobs that are not straight segments.
    triangles = [(col, cells) for col, cells in components(B, False) if not is_straight(cells)]

    ops, sels = [], []

    # 1) Erase all triangle marker objects in one stroke (union of the marker objects in I).
    tri_cells = set()
    for _, cells in triangles:
        tri_cells |= {back(rc) for rc in cells}
    if tri_cells:
        ops.append(bgc)
        sels.append(sel_of(tri_cells))

    # 2) Each marker shoots a new full diagonal in its guide line's colour, starting just
    #    past the marker's tip; all new lines of the same colour are drawn in one stroke.
    by_col = {}
    for _, cells in triangles:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if (r0, c1) in cells:
            corner, off, anchor = (r0, c1), (-1, 1), (r0 + 1, c0)
        else:
            corner, off, anchor = (r1, c0), (1, -1), (r0, c0 + 1)
        line_col = int(B[anchor[0], anchor[1]])
        d = (corner[0] + off[0]) - (corner[1] + off[1])
        new_cells = [back((r, r - d)) for r in range(h) if 0 <= r - d < w]
        if new_cells:
            by_col.setdefault(line_col, set()).update(new_cells)
    for col in sorted(by_col):
        ops.append(col)
        sels.append(sel_of(by_col[col]))

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
