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
import numpy as np
from maker.sel_helpers import sel_of

def _unifint(lb, ub, bounds):
    a, b = bounds
    return min(max(a, round(a + (b - a) * random.uniform(lb, ub))), b)

def sample_colors(num_examples=None) -> dict:
    c1, c2, c3, c4 = random.sample(range(10), 4)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}

def _components(g):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            col = g[i, j]
            stack = [(i, j)]; seen[i, j] = True; cells = []
            while stack:
                r, c = stack.pop(); cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and g[rr, cc] == col:
                        seen[rr, cc] = True; stack.append((rr, cc))
            comps.append(cells)
    return comps

def _nbrs(r, c, h, w):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w]

def _diag(r, c, h, w):
    out = []
    for dr, dc in ((1, 1), (-1, -1), (-1, 1), (1, -1)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc)); rr += dr; cc += dc
    return out

def generate(diff_lb, diff_ub, max_h, max_w, c1, c2, c3, c4, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = np.zeros((h, w), int)
    magiccol = 0
    ndivi = _unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    for _ in range(ndivi):
        objs = []
        for cells in _components(gi):
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            oh, ow = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            if min(oh, ow) > 3 and max(oh, ow) > 4 and oh * ow == len(cells):
                objs.append((min(rs), min(cs), max(rs), max(cs)))
        if not objs:
            break
        t, l, b, r = random.choice(objs)
        magiccol += 1
        if random.choice((True, False)):
            loci = random.randint(t + 2, b - 1)
            gi[loci:b + 1, l:r + 1] = magiccol
        else:
            locj = random.randint(l + 2, r - 1)
            gi[t:b + 1, locj:r + 1] = magiccol
    comps = _components(gi)
    random.shuffle(comps)
    g = np.zeros((h, w), int)
    for ii, cells in enumerate(comps):
        col = c1 if ii == 0 else (c2 if ii == 1 else random.choice((c1, c2)))
        for rr, cc in cells:
            g[rr, cc] = col
    ofc1 = {(r, c) for r in range(h) for c in range(w) if g[r, c] == c1}
    ofc2 = {(r, c) for r in range(h) for c in range(w) if g[r, c] == c2}
    mn = min(len(ofc1), len(ofc2))
    n1 = _unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    n2 = _unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    def pick(ofc, n):
        srcs = set()
        for _ in range(n):
            blocked = set(srcs)
            for (r, c) in srcs:
                blocked |= set(_nbrs(r, c, h, w))
            cands = sorted(ofc - blocked)
            if not cands:
                break
            srcs.add(random.choice(cands))
        return srcs
    s1 = pick(ofc1, n1); s2 = pick(ofc2, n2)
    for (r, c) in s1: g[r, c] = c3
    for (r, c) in s2: g[r, c] = c4
    lns = set()
    for (r, c) in s1 | s2:
        lns |= set(_diag(r, c, h, w))
    go = g.copy()
    for (r, c) in ofc1 & lns: go[r, c] = c3
    for (r, c) in ofc2 & lns: go[r, c] = c4
    return {"input": g.tolist(), "output": go.tolist()}

def _markers(G):
    # marker colours: no cell of that colour has a same-coloured 8-neighbour
    h, w = G.shape
    cols = sorted(set(G.flatten().tolist()))
    ms = []
    for col in cols:
        ok = True
        for r in range(h):
            for c in range(w):
                if G[r, c] == col and any(G[a, b] == col for a, b in _nbrs(r, c, h, w)):
                    ok = False; break
            if not ok: break
        if ok: ms.append(col)
    return ms

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    mk = _markers(I)
    regions = [c for c in sorted(set(I.flatten().tolist())) if c not in mk]
    # marker colour -> region colour it sits in, by neighbour counts over I and demo inputs
    grids = [I] + [np.asarray(e[0], dtype=int) for e in (examples or [])]
    score = {}
    for G in grids:
        gh, gw = G.shape
        for m in mk:
            nb = set()
            for r in range(gh):
                for c in range(gw):
                    if G[r, c] == m:
                        nb |= set(_nbrs(r, c, gh, gw))
            for (a, b) in nb:
                k = (m, int(G[a, b]))
                score[k] = score.get(k, 0) + 1
    mapping = {}  # region colour -> marker colour
    if len(mk) == 2 and len(regions) == 2:
        m1, m2 = mk; r1, r2 = regions
        s_a = score.get((m1, r1), 0) + score.get((m2, r2), 0)
        s_b = score.get((m1, r2), 0) + score.get((m2, r1), 0)
        mapping = {r1: m1, r2: m2} if s_a >= s_b else {r1: m2, r2: m1}
    else:
        for m in mk:
            best = max(regions, key=lambda rc: score.get((m, rc), 0)) if regions else None
            if best is not None and best not in mapping:
                mapping[best] = m
    G = I.copy()
    ops, sels = [], []
    srcs = [(r, c) for r in range(h) for c in range(w) if I[r, c] in mk]
    for (r, c) in srcs:
        # shoot the 4 diagonal rays from this marker; recolour region cells they cross
        ray = _diag(r, c, h, w)
        for rc, mc in mapping.items():
            cells = [(a, b) for (a, b) in ray if G[a, b] == rc]
            if cells:
                ops.append(int(mc)); sels.append(sel_of(cells))
                for (a, b) in cells:
                    G[a, b] = mc
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
