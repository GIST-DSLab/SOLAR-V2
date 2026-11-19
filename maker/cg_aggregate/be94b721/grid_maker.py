"""
ARC Task: be94b721 (RE-ARC) — LLM-generated grid_maker
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
    # Only the canvas colour is fixed for the episode. The target colour and the
    # distractor colours are drawn per instance; the rule is "largest object",
    # so it does not depend on which colour the object has.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if a >= b:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def neigh(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)}

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    no = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    nc = unifint(diff_lb, diff_ub, (no + 1, max(no + 1, 2 * no)))

    gi = np.full((h, w), bgc, dtype=int)
    inds = {(r, c) for r in range(h) for c in range(w)}
    ch = random.choice(sorted(inds))
    shp = {ch}
    inds.discard(ch)
    for _ in range(nc - 1):
        border = set()
        for p in shp:
            border |= dneigh(p)
        cands = sorted((inds - shp) & border)
        if not cands:
            break
        shp.add(random.choice(cands))
    allnb = set()
    for p in shp:
        allnb |= neigh(p)
    inds = (inds - shp) - allnb
    trgc = random.choice(remcols)
    for (r, c) in shp:
        gi[r, c] = trgc
    rs = [p[0] for p in shp]
    cs = [p[1] for p in shp]
    r0, c0 = min(rs), min(cs)
    go = np.full((max(rs) - r0 + 1, max(cs) - c0 + 1), bgc, dtype=int)
    for (r, c) in shp:
        go[r - r0, c - c0] = trgc
    size = len(shp)

    for _ in range(no):
        if not inds:
            break
        ch = random.choice(sorted(inds))
        s2 = {ch}
        nc2 = unifint(diff_lb, diff_ub, (1, max(1, size - 1)))
        for _ in range(nc2 - 1):
            border = set()
            for p in s2:
                border |= dneigh(p)
            cands = sorted((inds - s2) & border)
            if not cands:
                break
            s2.add(random.choice(cands))
        col = random.choice(remcols)
        for (r, c) in s2:
            gi[r, c] = col
        nb = set()
        for p in s2:
            nb |= neigh(p)
        inds = (inds - s2) - nb

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: the answer is the single largest one-colour object, cut out on its
    # own background.
    # The golf entry treats "all cells of the dominant colour" as one unit and
    # keeps the rows and columns that contain it. Keeping those rows and columns
    # is the same as cropping to the object's bounding box, so the two readings
    # agree on the crop.
    # They disagree on what counts as the unit. This generator can give a
    # distractor the same colour as the target, so "all cells of that colour" is
    # not reliable here. The unit stays the connected object, as in the maker.
    # Other objects' cells can fall inside the bounding box. The golf entry
    # leaves them in place; the generator's output paints them over with
    # background, so they are cleared first.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background is the canvas colour, which is the same in every demonstration.
    # Choose the colour that is the majority in I and in the example inputs.
    cnt = Counter(I.flatten().tolist())
    if examples:
        tot = Counter()
        for ex in examples:
            xi = np.asarray(ex[0], dtype=int)
            tot[Counter(xi.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = tot.most_common(1)[0][0]
        bgc = cand if cand in cnt else cnt.most_common(1)[0][0]
    else:
        bgc = cnt.most_common(1)[0][0]

    # Find the largest 4-connected one-colour object that is not background.
    seen = np.zeros((hi, wi), bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                cr, cc = dq.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            if best is None or len(comp) > len(best):
                best = comp

    comp = best
    objset = set(comp)
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []

    # 1. Clear the cells of other objects that lie inside the target's bounding
    #    box. These are non-background cells that are not part of the target.
    contam = [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
              if (rr, cc) not in objset and int(I[rr, cc]) != bgc]
    if contam:
        ops.append(int(bgc)); sels.append(sel_of(contam))

    # 2. Crop to the object's bounding box. These are exactly the rows and
    #    columns that contain the object, and the whole rectangle is meant.
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])

    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task be94b721"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task be94b721"
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
                                f"for task be94b721"
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
                    f"Failed to build a complete episode for task be94b721 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"be94b721-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
