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
from collections import Counter, deque
import numpy as np
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
    return {"bgc": bgc, "instance_plan": ex + [dict(random.choice(ex))]}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None) -> dict:
    def unifint(lo, hi):
        a, b = lo, hi
        if b < a:
            b = a
        d = random.uniform(diff_lb, diff_ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if mirror is None:
        mirror = random.choice((True, False))
    h = unifint(6, max(6, max_h))
    w = unifint(6, max(6, max_w))
    nlns = unifint(1, max(1, (h + w) // 8))
    remcols = [c for c in range(10) if c != bgc]
    full = {(i, j) for i in range(h) for j in range(w)}
    inds = set(full)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def fill(g, col, cells):
        for i, j in cells:
            if 0 <= i < h and 0 <= j < w:
                g[i][j] = col

    def nbrs(cells):
        out = set()
        for i, j in cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    out.add((i + di, j + dj))
        return out

    spopts = [(i, 0) for i in range(h - 5, -1, -1)] + [(0, j) for j in range(1, w - 4)]
    succ, tr, maxtr = 0, 0, 10 * nlns
    while succ < nlns and tr < maxtr:
        tr += 1
        if not spopts:
            break
        sp = random.choice(spopts)
        ln = {(sp[0] + k, sp[1] + k) for k in range(max(h, w))} & full
        if not ln.issubset(inds):
            continue
        lno = sorted(ln)
        trid1 = random.randint(2, min(5, len(lno) - 3))
        trid2 = random.randint(2, min(5, len(lno) - 3))
        tri1 = {(i, j) for i in range(trid1) for j in range(trid1) if j >= i}
        a = random.choice(lno[1:-trid1 - 1]); triloc1 = (a[0], a[1] + 1)
        tri2 = {(j, i) for i in range(trid2) for j in range(trid2) if j >= i}
        b = random.choice(lno[1:-trid2 - 1]); triloc2 = (b[0] + 1, b[1])
        spo2 = (sp[0], sp[1] - trid2 - 2)
        nex2 = {(spo2[0] + k, spo2[1] + k) for k in range(max(h, w))} & full
        spo1 = (sp[0] - trid1 - 2, sp[1])
        nex1 = {(spo1[0] + k, spo1[1] + k) for k in range(max(h, w))} & full
        parts = random.sample([(tri1, triloc1, nex1), (tri2, triloc2, nex2)], 2)
        lncol = None
        fullobj = None
        for idx, (tri, tl, nex) in enumerate(parts):
            tri = {(i + tl[0], j + tl[1]) for i, j in tri}
            fullobj = ln | tri | nex
            if idx == 0:
                lncol, tricol = random.sample(remcols, 2)
            else:
                tricol = random.choice([c for c in remcols if c != lncol])
            ok = fullobj.issubset(inds) if idx == 0 else (tri | nex).issubset(fullobj)
            if ok and any(go[i][j] not in (bgc, lncol) or gi[i][j] != bgc for i, j in (ln | nex) - ln):
                ok = False  # a new diagonal would land on another line's cells: ambiguous paint order
            if ok and any(gi[i][j] != bgc or go[i][j] != bgc for i, j in tri | ln):
                ok = False  # marker/line would cover cells already drawn
            if ok:
                succ += 1
                inds = (inds - fullobj) - nbrs(fullobj)
                fill(gi, tricol, tri); fill(gi, lncol, ln)
                fill(go, lncol, ln); fill(go, lncol, nex)
    if mirror:
        gi = gi[::-1]; go = go[::-1]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    A = np.asarray(I, dtype=int)
    h, w = A.shape
    # background: the canvas colour every demonstration is painted on
    bgc = None
    if examples:
        cands = None
        for ei, _ in examples:
            E = np.asarray(ei, dtype=int)
            mc = Counter(E.flatten().tolist()).most_common(1)[0][0]
            cands = {mc} if cands is None else cands & {mc}
        if cands and len(cands) == 1:
            bgc = next(iter(cands))
    if bgc is None:
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
                col = int(G[r, c]); q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True; q.append((ny, nx))
                out.append((col, set(cells)))
        return out

    def is_dr(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs); n = max(rs) - r0
        return n == max(cs) - c0 and cells == {(r0 + k, c0 + k) for k in range(n + 1)}

    def is_straight(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            return cells == {(r0, c) for c in range(c0, c1 + 1)}
        if c0 == c1:
            return cells == {(r, c0) for r in range(r0, r1 + 1)}
        return is_dr(cells)

    # Orientation: guide lines run down-right in canonical form, else the scene is mirrored.
    flip = not any(len(cells) >= 3 and is_dr(cells) for _, cells in components(A, True))
    B = A[::-1, :].copy() if flip else A

    def back(rc):
        return (h - 1 - rc[0], rc[1]) if flip else rc

    cur = A.copy()
    markers = [(col, cells) for col, cells in components(B, False) if not is_straight(cells)]
    ops, sels = [], []

    # Reading (golf order): each marker is a pointer. 1) First shoot a parallel diagonal,
    # in its guide line's colour, through the cell just beyond the marker's far corner.
    for col, cells in markers:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if (r0, c1) in cells:
            corner, off, anchor = (r0, c1), (-1, 1), (r0 + 1, c0)
        else:
            corner, off, anchor = (r1, c0), (1, -1), (r0, c0 + 1)
        line_col = int(B[anchor])
        d = (corner[0] + off[0]) - (corner[1] + off[1])
        new_cells = [back((r, r - d)) for r in range(h) if 0 <= r - d < w]
        # skip only when the grid as it stands already shows this diagonal in that colour
        if not new_cells or all(int(cur[r, c]) == line_col for r, c in new_cells):
            continue
        ops.append(line_col); sels.append(sel_of(new_cells))
        for r, c in new_cells:
            cur[r, c] = line_col

    # 2) Then remove the pointer markers, each as one whole object (its own cells from I;
    #    a flood fill could leak into a just-drawn diagonal of the same colour).
    for col, cells in markers:
        ops.append(bgc); sels.append(sel_of([back(x) for x in sorted(cells)]))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
