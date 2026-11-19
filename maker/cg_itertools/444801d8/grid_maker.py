"""
ARC Task: 444801d8 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    return {"bgc": bgc, "ccols": random.sample(rem, numcols)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))

    if ccols is None:
        ccols = random.sample([c for c in range(10) if c != bgc], 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(4, 6)
        bx = {(r, c) for r in range(1, oh) for c in range(5)
              if r in (1, oh - 1) or c in (0, 4)} - {(1, 2)}
        full = {(r, c) for r in range(oh) for c in range(5)}
        cands = sorted({(r, c) for r in range(1, oh) for c in range(5)} - bx)
        dot = random.choice(cands)
        dcol, bxcol = random.sample(ccols, 2)
        inobj = {(bxcol, p) for p in bx} | {(dcol, dot)}
        outobj = {(bxcol, p) for p in bx} | {(dcol, p) for p in full - bx}
        if random.choice((True, False)):
            # hmirror within own bbox (rows 1..oh-1), then shift UP
            inobj = {(v, (oh - 1 - r, c)) for v, (r, c) in inobj}
            outobj = {(v, (oh - 1 - r, c)) for v, (r, c) in outobj}
        lc = [p for p in inds if p[0] <= h - oh and p[1] <= w - 5]
        if not lc:
            continue
        li, lj = random.choice(sorted(lc))
        outp = {(v, (r + li, c + lj)) for v, (r, c) in outobj}
        outi = {p for _, p in outp}
        if outi <= inds:
            succ += 1
            inp = {(v, (r + li, c + lj)) for v, (r, c) in inobj}
            ii = {p for _, p in inp}
            r0 = min(r for r, _ in ii); r1 = max(r for r, _ in ii)
            c0 = min(c for _, c in ii); c1 = max(c for _, c in ii)
            ob = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)
                  if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)}
            inds = (inds - outi) - ob
            for v, (r, c) in inp:
                if 0 <= r < h and 0 <= c < w:
                    gi[r][c] = v
            for v, (r, c) in outp:
                if 0 <= r < h and 0 <= c < w:
                    go[r][c] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Where this agrees with the maker: both read the same rule. Each multi-cell
    object is a box with one gap. Its single dot's colour fills the box's hollow
    and also forms a 5-wide band on the row just outside the gap. Nothing
    changes there.

    Where it differs: the golf entry spreads the dot's colour step by step
    through background cells, so the dye moves outward from the dot like a
    flood. The maker paints a precomputed "bbox minus outline" mask instead.
    This version follows the golf entry's reading:
      (1) paint the band across the row outside the gap with the dot colour.
          The band closes the gap, so the box becomes a sealed container.
      (2) pour the colour in with FloodFill, one fill per background pocket of
          the sealed hollow. The flood reaches only the hollow because the band
          now blocks the gap. There is normally one pocket; there are two when
          the dot sits right behind the gap and cuts the gap cell off from the
          interior.
    Everything is measured from I. bgc comes from I plus the demonstration
    inputs.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt_bg = Counter(I.flatten().tolist())
    for ex in (examples or []):
        cnt_bg.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt_bg.most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            comps.append((col, cells))
    boxes = [t for t in comps if len(t[1]) > 1]
    boxes.sort(key=lambda t: (min(r for r, _ in t[1]), min(c for _, c in t[1])))

    ops, sels = [], []
    for col, cells in boxes:
        cs_ = set(cells)
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        hollow = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if (r, c) not in cs_]
        if not hollow:
            continue
        # dot colour = least-common colour inside the hollow
        hc = Counter(int(I[r, c]) for r, c in hollow)
        dcol = min(hc.items(), key=lambda kv: (kv[1], kv[0]))[0]
        gap_on_top = any(r == r0 for r, _ in hollow)
        line_r = r0 - 1 if gap_on_top else r1 + 1

        # (1) band of the dot colour just outside the gap; this closes the box
        if 0 <= line_r < h:
            ops.append(dcol)
            sels.append(sel_of([(line_r, c) for c in range(c0, c1 + 1)]))

        # (2) pour the colour in: one FloodFill per background pocket of the hollow
        left = {p for p in hollow if I[p] == bgc}
        while left:
            seed = min(left, key=lambda p: (p[0] in (r0, r1), p))  # prefer interior
            comp, st = {seed}, [seed]
            while st:
                y, x = st.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (y + dy, x + dx)
                    if q in left and q not in comp:
                        comp.add(q); st.append(q)
            left -= comp
            ops.append(10 + dcol)
            sels.append([seed[0], seed[1], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 444801d8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 444801d8"
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
                                f"for task 444801d8"
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
                    f"Failed to build a complete episode for task 444801d8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"444801d8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
