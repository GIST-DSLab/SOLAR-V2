"""
ARC Task: d07ae81c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    c1, c2, c3, c4 = random.sample(range(10), 4)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, c1, c2, c3, c4) -> dict:
    def unifint(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + int(round(d * (hi - lo)))))

    def comps(g):
        H, W = len(g), len(g[0])
        seen = set()
        out = []
        for r in range(H):
            for c in range(W):
                if (r, c) in seen:
                    continue
                col = g[r][c]
                st = [(r, c)]
                seen.add((r, c))
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and (x, y) not in seen and g[x][y] == col:
                            seen.add((x, y))
                            st.append((x, y))
                out.append(cells)
        return out

    def diag_nbrs(r, c, H, W):
        return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                if (a or b) and 0 <= r + a < H and 0 <= c + b < W}

    h = unifint(10, max(10, min(30, max_h)))
    w = unifint(10, max(10, min(30, max_w)))
    gi = [[0] * w for _ in range(h)]
    ndivi = unifint(1, (h * w) // 10)
    magic = 0
    for _ in range(ndivi):
        cand = []
        for o in comps(gi):
            rs = [p[0] for p in o]; cs = [p[1] for p in o]
            oh = max(rs) - min(rs) + 1; ow = max(cs) - min(cs) + 1
            if min(oh, ow) > 3 and max(oh, ow) > 4 and oh * ow == len(o):
                cand.append((min(rs), min(cs), max(rs), max(cs)))
        if not cand:
            break
        t, l, b, rr = random.choice(cand)
        if random.choice((True, False)):
            r0, c0 = random.randint(t + 2, b - 1), l
        else:
            r0, c0 = t, random.randint(l + 2, rr - 1)
        magic += 1
        for x in range(r0, b + 1):
            for y in range(c0, rr + 1):
                gi[x][y] = magic
    objs = comps(gi)
    random.shuffle(objs)
    for ii, o in enumerate(objs):
        col = c1 if ii == 0 else (c2 if ii == 1 else random.choice((c1, c2)))
        for (x, y) in o:
            gi[x][y] = col
    ofc1 = {(x, y) for x in range(h) for y in range(w) if gi[x][y] == c1}
    ofc2 = {(x, y) for x in range(h) for y in range(w) if gi[x][y] == c2}
    mn = min(len(ofc1), len(ofc2))
    n1 = unifint(1, max(1, int(mn ** 0.5)))
    n2 = unifint(1, max(1, int(mn ** 0.5)))

    def pick(ofc, n):
        srcs = set()
        for _ in range(n):
            blocked = set()
            for s in srcs:
                blocked |= diag_nbrs(s[0], s[1], h, w)
            cands = sorted((ofc - srcs) - blocked)
            if not cands:
                break
            srcs.add(random.choice(cands))
        return srcs

    s1 = pick(ofc1, n1)
    s2 = pick(ofc2, n2)
    for (x, y) in s1:
        gi[x][y] = c3
    for (x, y) in s2:
        gi[x][y] = c4
    lns = set()
    for (x, y) in s1 | s2:
        for dr, dc in ((1, 1), (-1, -1), (-1, 1), (1, -1)):
            a, b = x, y
            while 0 <= a < h and 0 <= b < w:
                lns.add((a, b)); a += dr; b += dc
    go = [row[:] for row in gi]
    for (x, y) in ofc1 & lns:
        go[x][y] = c3
    for (x, y) in ofc2 & lns:
        go[x][y] = c4
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    G = I.copy()
    ops, sels = [], []

    def nbrs(r, c):
        return [(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                if (a or b) and 0 <= r + a < h and 0 <= c + b < w]

    cells_by_color = {}
    for r in range(h):
        for c in range(w):
            cells_by_color.setdefault(int(I[r, c]), []).append((r, c))

    # marker colours: every cell has no same-colour 8-neighbour
    markers = []
    for col, cells in cells_by_color.items():
        s = set(cells)
        if all(not any(p in s for p in nbrs(r, c)) for r, c in cells):
            markers.append(col)

    # which region colour each marker colour recolours: read it from the
    # demonstrations (region colour -> marker colour changes); fall back to
    # the majority colour around the markers.
    paint = {}
    for ex in (examples or []):
        EI = np.asarray(ex[0], dtype=int); EO = np.asarray(ex[1], dtype=int)
        if EI.shape != EO.shape:
            continue
        diff = EI != EO
        for a, b in zip(EI[diff].tolist(), EO[diff].tolist()):
            if b in markers and a not in markers:
                paint[int(a)] = int(b)
    if len(paint) < 2:
        paint = {}
        for m in markers:
            cnt = Counter()
            for (r, c) in cells_by_color[m]:
                for (a, b) in nbrs(r, c):
                    if int(I[a, b]) not in markers:
                        cnt[int(I[a, b])] += 1
            if cnt:
                paint[cnt.most_common(1)[0][0]] = m

    marker_cells = [p for m in markers for p in cells_by_color[m]]

    # Unit of work: one whole diagonal LINE of the grid passing through a marker.
    # On each line, recolour all its cells of region colour X to X's marker
    # colour in one Color op (one op per region colour present on the line).
    lines = []
    for k in sorted({r - c for r, c in marker_cells}):
        lines.append([(r, r - k) for r in range(h) if 0 <= r - k < w])
    for k in sorted({r + c for r, c in marker_cells}):
        lines.append([(r, k - r) for r in range(h) if 0 <= k - r < w])

    for line in lines:
        for region in sorted(paint):
            cells = [(r, c) for r, c in line if G[r, c] == region]
            if not cells:
                continue
            ops.append(int(paint[region])); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = paint[region]

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
                        f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
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
                                f"for task d07ae81c"
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
                    f"Failed to build a complete episode for task d07ae81c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d07ae81c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
