"""
ARC Task: d22278a0 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors / plan
def sample_colors(num_examples=None) -> dict:
    """Fix background + the colour of each of the 4 grid corners for the whole episode,
    and pre-plan how many corners are marked in each instance (structural variant)."""
    bgc = random.choice(range(10))
    # corner colours are always NON-ZERO: 0 is 'transparent' for Copy/Paste in ARCLE
    pool = [c for c in range(1, 10) if c != bgc]
    random.shuffle(pool)
    ccols = pool[:4]

    n_ex = num_examples if num_examples else 3
    variants = [1, 2, 3, 4]                      # number of marked corners
    if n_ex >= len(variants):
        plan = [{"ncorns": v} for v in variants]
        plan += [{"ncorns": random.choice(variants)} for _ in range(n_ex - len(variants))]
        random.shuffle(plan)
    else:
        plan = [{"ncorns": v} for v in random.sample(variants, n_ex)]
    plan = plan + [dict(random.choice(plan))]    # test case is one of the shown variants
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols,
             ncorns=None, instance_plan=None, **kwargs) -> dict:
    def unif(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(lo, min(hi, b))
        return random.randint(lo, hi)

    if ncorns is None:
        ncorns = random.randint(1, 4)
    ncorns = max(1, min(4, int(ncorns)))

    gi = go = None
    for _attempt in range(40):
        hlo = min(4 + 2 * ncorns, max_h)
        wlo = min(4 + 2 * ncorns, max_w)
        h = max(4, unif(diff_lb, diff_ub, (hlo, max_h)))
        w = max(4, unif(diff_lb, diff_ub, (wlo, max_w)))

        corner_pts = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        idxs = random.sample(range(4), ncorns)
        chosen = [(corner_pts[i], ccols[i]) for i in idxs]

        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        for (p, col) in chosen:
            gi[p[0]][p[1]] = col
            go[p[0]][p[1]] = col

        for (p, col) in chosen:
            pr, pc = p
            for r in range(h):
                for c in range(w):
                    if max(abs(r - pr), abs(c - pc)) % 2:      # even Chebyshev rings only
                        continue
                    d1 = abs(r - pr) + abs(c - pc)
                    others = [abs(r - q[0]) + abs(c - q[1]) for (q, _) in chosen if q != p]
                    if others and d1 >= min(others):           # strict Manhattan-nearest
                        continue
                    go[r][c] = col

        if go != gi:
            break

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ---------------------------------------------------------------- trajectory
def _split_runs(vals):
    vals = sorted(set(vals))
    runs, s, p = [], vals[0], vals[0]
    for v in vals[1:]:
        if v == p + 1:
            p = v
        else:
            runs.append((s, p))
            s = p = v
    runs.append((s, p))
    return runs


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bg = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- the marked grid corners (the anchors the rule radiates from)
    marks, seen = [], set()
    for p in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if p in seen:
            continue
        seen.add(p)
        if int(I[p]) != bg:
            marks.append((p, int(I[p])))

    # --- rule: every cell at EVEN Chebyshev distance from a corner, inside that
    #     corner's strict Manhattan-Voronoi region, takes the corner's colour.
    targets = {}
    for (p, col) in marks:
        pr, pc = p
        for r in range(h):
            for c in range(w):
                if max(abs(r - pr), abs(c - pc)) % 2:
                    continue
                d1 = abs(r - pr) + abs(c - pc)
                others = [abs(r - q[0]) + abs(c - q[1]) for (q, _) in marks if q != p]
                if others and d1 >= min(others):
                    continue
                targets[(r, c)] = col

    ok = all(targets.get((r, c), bg) == int(O[r, c]) for r in range(h) for c in range(w))
    if not ok:  # defensive fallback
        targets = {(r, c): int(O[r, c]) for r in range(h) for c in range(w) if O[r, c] != bg}

    ops, sels = [], []
    clip = None   # (colour, orientation, length) currently on the clipboard

    def emit_run(col, orient, fixed, run, drawn):
        """Draw one straight stripe of this corner's pattern.
        If a stripe of this corner already drawn fits inside it, REPLICATE it
        (CopyO + Paste at the new offset) and only paint what the rule leaves over."""
        nonlocal clip
        a, b = run
        best = None
        for (f2, a2, b2) in drawn:
            L = b2 - a2 + 1
            if L >= 2 and a2 >= a and b2 <= b:
                key = (L, 1 if clip == (col, orient, L) else 0)
                if best is None or key > best[0]:
                    best = (key, (f2, a2, b2))
        if best is not None and col != 0:
            f2, a2, b2 = best[1]
            L = b2 - a2 + 1
            if clip != (col, orient, L):
                ops.append(29)                                    # CopyO the source stripe
                sels.append([f2, a2, 0, L - 1] if orient == 'H' else [a2, f2, L - 1, 0])
                clip = (col, orient, L)
            ops.append(30)                                        # Paste it at this offset
            sels.append([fixed, a2, 0, 0] if orient == 'H' else [a2, fixed, 0, 0])
            rest = [x for x in range(a, b + 1) if not (a2 <= x <= b2)]
            if rest:
                cells = [(fixed, x) if orient == 'H' else (x, fixed) for x in rest]
                ops.append(col)
                sels.append(sel_of(cells))
        else:
            cells = [(fixed, x) if orient == 'H' else (x, fixed) for x in range(a, b + 1)]
            ops.append(col)
            sels.append(sel_of(cells))
        drawn.append((fixed, a, b))

    for (p, col) in marks:
        pr, pc = p
        cells = [rc for rc, cc in targets.items() if cc == col and rc != (pr, pc)]
        if not cells:
            continue
        Hg, Vg = {}, {}
        for (r, c) in cells:
            dr, dc = abs(r - pr), abs(c - pc)
            d = max(dr, dc)
            if dr >= dc:
                Hg.setdefault(d, []).append((r, c))   # horizontal stripes
            else:
                Vg.setdefault(d, []).append((r, c))   # vertical stripes

        drawn_h = []
        for d in sorted(Hg):                          # outward from the corner
            for row in sorted({r for (r, _) in Hg[d]}):
                cols = [c for (r, c) in Hg[d] if r == row]
                for run in _split_runs(cols):
                    emit_run(col, 'H', row, run, drawn_h)

        drawn_v = []
        for d in sorted(Vg):                          # outward from the corner
            for cc in sorted({c for (_, c) in Vg[d]}):
                rows = [r for (r, c) in Vg[d] if c == cc]
                for run in _split_runs(rows):
                    emit_run(col, 'V', cc, run, drawn_v)

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
                        f"num_examples+1 ({num_examples + 1}) for task d22278a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d22278a0"
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
                                f"for task d22278a0"
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
                    f"Failed to build a complete episode for task d22278a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d22278a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
