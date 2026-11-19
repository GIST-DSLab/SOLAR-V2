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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * min(lb, ub)))
    hi = a + int(round((b - a) * max(lb, ub)))
    return random.randint(lo, max(lo, hi))


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, mirror=None, **kw) -> dict:
    if mirror is None:
        mirror = random.choice((True, False))
    remcols = [c for c in range(10) if c != bgc]
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    nlns = _unifint(diff_lb, diff_ub, (1, max(1, (h + w) // 8)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    fullinds = {(i, j) for i in range(h) for j in range(w)}
    inds = set(fullinds)

    def neighbors(c):
        r, cc = c
        return {(r + a, cc + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    spopts = [(idx, 0) for idx in range(h - 5, -1, -1)] + [(0, idx) for idx in range(1, w - 4)]
    succ, tr, maxtr = 0, 0, 10 * nlns
    while succ < nlns and tr < maxtr:
        tr += 1
        if not spopts:
            break
        sp = random.choice(spopts)
        ln = {(sp[0] + k, sp[1] + k) for k in range(max(h, w))} & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln)
        t1 = random.randint(2, min(5, len(lno) - 3))
        t2 = random.randint(2, min(5, len(lno) - 3))
        tri1 = {(i, j) for i in range(t1) for j in range(t1) if j >= i}
        p = random.choice(lno[1:-t1 - 1]); triloc1 = (p[0], p[1] + 1)
        tri2 = {(j, i) for i in range(t2) for j in range(t2) if j >= i}
        p = random.choice(lno[1:-t2 - 1]); triloc2 = (p[0] + 1, p[1])
        spo2 = (sp[0], sp[1] - t2 - 2)
        nex2 = {(spo2[0] + k, spo2[1] + k) for k in range(max(h, w))} & fullinds
        spo1 = (sp[0] - t1 - 2, sp[1])
        nex1 = {(spo1[0] + k, spo1[1] + k) for k in range(max(h, w))} & fullinds
        t1s = {(a + triloc1[0], b + triloc1[1]) for a, b in tri1}
        t2s = {(a + triloc2[0], b + triloc2[1]) for a, b in tri2}
        # keep each line's whole unit (line, both markers, both new lines) clear of
        # everything placed before, so no two drawings overwrite each other
        if not (ln | t1s | t2s | nex1 | nex2).issubset(inds):
            continue
        items = [(tri1, triloc1, nex1), (tri2, triloc2, nex2)]
        random.shuffle(items)
        lncol = None
        for idx, (tri, triloc, nex) in enumerate(items):
            tri = {(a + triloc[0], b + triloc[1]) for a, b in tri}
            fullobj = ln | tri | nex
            if idx == 0:
                lncol, tricol = random.sample(remcols, 2)
            else:
                tricol = random.choice([c for c in remcols if c != lncol])
            if (fullobj.issubset(inds) if idx == 0 else (tri | nex).issubset(fullobj)):
                succ += 1
                nb = set()
                for c in fullobj:
                    nb |= neighbors(c)
                inds = (inds - fullobj) - nb
                for a, b in tri:
                    gi[a][b] = tricol
                for a, b in ln:
                    gi[a][b] = lncol
                    go[a][b] = lncol
                for a, b in nex:
                    go[a][b] = lncol
    if mirror:
        gi = gi[::-1]
        go = go[::-1]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading shared with the golf entry (and differing from the maker's): a guide
    # line and the staircase markers hugging it form one unit; each marker's SIZE
    # t (its run along the diagonal touching the line) fixes a new copy of the line
    # at diagonal offset t+2 on the marker's side. Markers are read per line from
    # the diagonals adjacent to it, not as free-floating 4-connected blobs (two
    # markers of the same colour from different lines may touch).
    A = np.asarray(I, dtype=int)
    h, w = A.shape
    if examples:
        bgs = [Counter(np.asarray(e[0]).flatten().tolist()).most_common(1)[0][0] for e in examples]
        bgc = Counter(bgs).most_common(1)[0][0]
    else:
        bgc = int(Counter(A.flatten().tolist()).most_common(1)[0][0])

    def comps8(G):
        H, W = G.shape
        seen = np.zeros((H, W), dtype=bool)
        out = []
        for r in range(H):
            for c in range(W):
                if seen[r, c] or G[r, c] == bgc:
                    continue
                col = G[r, c]
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and G[ny, nx] == col:
                                seen[ny, nx] = True; q.append((ny, nx))
                out.append((int(col), set(cells)))
        return out

    def guide_lines(G):
        res = []
        for col, cells in comps8(G):
            if len(cells) < 3:
                continue
            ds = {r - c for r, c in cells}
            if len(ds) != 1:
                continue
            d = ds.pop()
            full = {(r, r - d) for r in range(h) if 0 <= r - d < w}
            if cells == full:
                res.append((col, d))
        return res

    lines = guide_lines(A)
    flip = False
    if not lines:
        flip = True
        lines = guide_lines(A[::-1, :])
    B = A[::-1, :] if flip else A

    def back(rc):
        r, c = rc
        return (h - 1 - r, c) if flip else (r, c)

    def run_along(dd, col):
        # contiguous run of marker cells on diagonal r-c=dd
        cells = [(r, r - dd) for r in range(h) if 0 <= r - dd < w]
        best = []
        cur = []
        for rc in cells:
            v = B[rc]
            if v != bgc and v != col:
                cur.append(rc)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = []
        if len(cur) > len(best):
            best = cur
        return best

    markers = []   # (cells of staircase, line colour, new diagonal)
    for col, d in lines:
        ur = run_along(d - 1, col)          # upper-right side, starts at line+(0,1)
        if len(ur) >= 2:
            t = len(ur); r0, c0 = ur[0]
            stair = [(r0 + i, c0 + j) for i in range(t) for j in range(i, t)]
            markers.append((stair, col, d - (t + 2)))
        ll = run_along(d + 1, col)          # lower-left side, starts at line+(1,0)
        if len(ll) >= 2:
            t = len(ll); r0, c0 = ll[0]
            stair = [(r0 + i, c0 + j) for i in range(t) for j in range(0, i + 1)]
            markers.append((stair, col, d + (t + 2)))

    ops, sels = [], []
    G = A.copy()
    # 1) erase every staircase marker (the whole object) to background
    for stair, col, nd in markers:
        cells = [back(rc) for rc in stair if 0 <= rc[0] < h and 0 <= rc[1] < w]
        cells = [rc for rc in cells if G[rc] != bgc]
        if not cells:
            continue
        ops.append(bgc); sels.append(sel_of(cells))
        for rc in cells:
            G[rc] = bgc
    # 2) for each marker, draw the line's copy at offset t+2 on the marker's side
    for stair, col, nd in markers:
        cells = [back((r, r - nd)) for r in range(h) if 0 <= r - nd < w]
        cells = [rc for rc in cells if G[rc] != col]
        if not cells:
            continue
        ops.append(col); sels.append(sel_of(cells))
        for rc in cells:
            G[rc] = col

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
