"""
ARC Task: 83302e8f (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or linc is None:
        cols = [c for c in range(10) if c not in (3, 4)]
        bgc, linc = random.sample(cols, 2)

    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (3, max(3, max_h // (h + 1))))
    nw = unifint(diff_lb, diff_ub, (3, max(3, max_w // (w + 1))))
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = np.full((fullh, fullw), bgc, dtype=int)
    for i in range(h, fullh, h + 1):
        gi[i, :] = linc
    for j in range(w, fullw, w + 1):
        gi[:, j] = linc

    def nbrs(r, c, H, W):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

    # crossing dots: line cells all of whose 4 neighbours (incl. off-grid) are line cells
    ofc = {(r, c) for r in range(fullh) for c in range(fullw) if gi[r, c] == linc}
    dots = set()
    for (r, c) in ofc:
        ok = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) not in ofc:
                ok = False
                break
        if ok:
            dots.add((r, c))
    tmp = gi.copy()
    for (r, c) in dots:
        tmp[r, c] = bgc
    # line segments (components of linc in tmp)
    seen = set()
    lns = []
    for (r, c) in ofc - dots:
        if (r, c) in seen:
            continue
        comp = []
        stack = [(r, c)]
        seen.add((r, c))
        while stack:
            p = stack.pop()
            comp.append(p)
            for q in nbrs(p[0], p[1], fullh, fullw):
                if q not in seen and tmp[q] == linc:
                    seen.add(q)
                    stack.append(q)
        lns.append(comp)
    cands = lns + [[d] for d in dots]
    nbreaks = unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    for obj in random.sample(cands, nbreaks):
        r, c = random.choice(obj)
        gi[r, c] = bgc

    # output
    go = gi.copy()
    seen = set()
    for r in range(fullh):
        for c in range(fullw):
            if gi[r, c] != bgc or (r, c) in seen:
                continue
            comp = []
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                p = stack.pop()
                comp.append(p)
                for q in nbrs(p[0], p[1], fullh, fullw):
                    if q not in seen and gi[q] == bgc:
                        seen.add(q)
                        stack.append(q)
            col = 3 if len(comp) == h * w else 4
            for p in comp:
                go[p] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    bgc = int(I[0, 0])  # (0,0) is always inside a lattice cell -> canvas colour

    # 4-connected components of the canvas colour in I
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc or seen[r, c]:
                continue
            comp = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                pr, pc = stack.pop()
                comp.append((pr, pc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < hi and 0 <= cc < wi and not seen[rr, cc] and I[rr, cc] == bgc:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(comp)

    ops, sels = [], []
    # Reading: every open canvas cell starts out "sealed" (3) ...
    all_bg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
    ops.append(3)
    sels.append(sel_of(all_bg))

    # ... then every region that leaks (not a solid rectangle of >1 cell:
    # merged through a broken line, or a lone hole in a crossing) floods to 4.
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        sealed = len(comp) == area and len(comp) > 1
        if not sealed:
            seed = min(comp)
            ops.append(14)  # FloodFill4 from one seed
            sels.append([seed[0], seed[1], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
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
                                f"for task 83302e8f"
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
                    f"Failed to build a complete episode for task 83302e8f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"83302e8f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
