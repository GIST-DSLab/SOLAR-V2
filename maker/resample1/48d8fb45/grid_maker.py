"""
ARC Task: 48d8fb45 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _neighbors(r, c):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    for _attempt in range(200):
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        h = min(h, max_h)
        w = min(w, max_w)
        nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
        tr = 0
        maxtr = 4 * nobjs
        done = False
        special_placed = False
        succ = 0
        gi = [[bgc] * w for _ in range(h)]
        inds = {(r, c) for r in range(h) for c in range(w)}
        go = None
        while tr < maxtr and succ < nobjs:
            oh = random.randint(2, 6)
            ow = random.randint(2, 6)
            bx = [(r, c) for r in range(oh) for c in range(ow)]
            nc = random.randint(3, oh * ow)
            sp = random.choice(bx)
            bxs = set(bx) - {sp}
            obj = {sp}
            for _ in range(nc - 1):
                cand = set()
                for (r, c) in obj:
                    cand |= set(_neighbors(r, c))
                cand = sorted((bxs - obj) & cand)
                obj.add(random.choice(cand))
            is_special = False
            if not done:
                done = True
                is_special = True
                idx = random.choice(sorted(obj))
                coll = random.choice(remcols)
                c3 = random.choice([c for c in remcols if c != coll])
                obj3 = [p for p in obj if p != idx]
                mr = min(r for r, _ in obj3)
                mc = min(c for _, c in obj3)
                Mr = max(r for r, _ in obj3)
                Mc = max(c for _, c in obj3)
                go = [[bgc] * (Mc - mc + 1) for _ in range(Mr - mr + 1)]
                for (r, c) in obj3:
                    go[r - mr][c - mc] = c3
                cobj = [(coll, idx)] + [(c3, p) for p in obj3]
            else:
                col = random.choice(remcols)
                cobj = [(col, p) for p in obj]
            locopts = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
            tr += 1
            if len(locopts) == 0:
                continue
            loc = random.choice(sorted(locopts))
            plcd = [(v, (r + loc[0], c + loc[1])) for v, (r, c) in cobj]
            plcdi = {p for _, p in plcd}
            if plcdi.issubset(inds):
                for v, (r, c) in plcd:
                    gi[r][c] = v
                succ += 1
                if is_special:
                    special_placed = True
                nb = set()
                for (r, c) in plcdi:
                    nb |= set(_neighbors(r, c))
                inds = (inds - plcdi) - nb
        if not special_placed or go is None:
            continue
        # verify background is the most common color
        flat = [v for row in gi for v in row]
        if Counter(flat).most_common(1)[0][0] != bgc:
            continue
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ny, nx in _neighbors(y, x):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, bgc)
    # the object carrying a single marker cell of a different colour
    best = max(comps, key=lambda cp: len(set(int(I[p]) for p in cp)))
    cnt = Counter(int(I[p]) for p in best)
    main = cnt.most_common(1)[0][0]
    body = [p for p in best if int(I[p]) == main]
    r0 = min(r for r, _ in body)
    c0 = min(c for _, c in body)
    r1 = max(r for r, _ in body)
    c1 = max(c for _, c in body)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # Crop canvas to the bounding box of the object's main-colour body
    ops.append(33); sels.append([r0, c0, bh - 1, bw - 1])
    # Current grid after transparent crop: zeros stay zero
    cur = I[r0:r1 + 1, c0:c1 + 1].copy()
    body_set = {(r - r0, c - c0) for r, c in body}
    # Everything inside the crop that is not the body (marker, intruding
    # neighbours, zero-holes when bgc != 0) becomes background
    clean = [(r, c) for r in range(bh) for c in range(bw)
             if (r, c) not in body_set and cur[r, c] != bgc]
    if clean:
        ops.append(bgc); sels.append(sel_of(clean))
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
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
                                f"for task 48d8fb45"
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
                    f"Failed to build a complete episode for task 48d8fb45 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"48d8fb45-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
