"""
ARC Task: 56dc2b01 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from maker.sel_helpers import sel_of


def _canvas(v, h, w):
    return [[v] * w for _ in range(h)]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(max(a, lo), min(b, hi))

    cols = [c for c in range(10) if c not in (2, 8)]
    if bgc is None or objc is None:
        bgc, objc = random.sample(cols, 2)
    max_h = max(4, min(30, max_h))
    max_w = max(6, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    oh = unifint(diff_lb, diff_ub, (1, h))
    ow = unifint(diff_lb, diff_ub, (1, max(1, (w - 1) // 2 - 1)))
    bb = [(i, j) for i in range(oh) for j in range(ow)]
    sp = random.choice(bb)
    obj = {sp}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    bbs = set(bb)
    for _ in range(ncells):
        cand = set()
        for (a, b) in obj:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if (da or db) and (a + da, b + db) in bbs:
                        cand.add((a + da, b + db))
        cand -= obj
        if not cand:
            break
        obj.add(random.choice(sorted(cand)))
    mi = min(a for a, b in obj); mj = min(b for a, b in obj)
    obj = {(a - mi, b - mj) for a, b in obj}
    oh = max(a for a, b in obj) + 1
    ow = max(b for a, b in obj) + 1
    loci = random.randint(0, h - oh)
    locj = unifint(diff_lb, diff_ub, (1, w - ow))
    gi = _canvas(bgc, h, w)
    barlocji = unifint(diff_lb, diff_ub, (0, locj))
    barlocj = locj - barlocji
    barlocj = min(max(0, barlocj), locj - 1)
    for r in range(h):
        gi[r][barlocj] = 2
    go = [row[:] for row in gi]
    for a, b in obj:
        go[a + loci][b + barlocj + 1] = objc
    lc = barlocj + ow + 1
    if lc < w:
        for r in range(h):
            go[r][lc] = 8
    for a, b in obj:
        gi[a + loci][b + locj] = objc
    gi = np.array(gi); go = np.array(go)

    fns = {
        "id": lambda g: g, "dm": lambda g: g.T, "cm": lambda g: np.rot90(g, 2).T,
        "vm": np.fliplr, "hm": np.flipud, "r90": lambda g: np.rot90(g, 3),
        "r180": lambda g: np.rot90(g, 2), "r270": lambda g: np.rot90(g, 1),
    }
    transposing = {"dm", "cm", "r90", "r270"}
    names = list(fns)
    nm = random.choice((1, 2))
    for _ in range(50):
        chosen = random.sample(names, nm)
        nt = sum(1 for n in chosen if n in transposing) % 2
        fh, fw = (w, h) if nt else (h, w)
        if fh <= max_h and fw <= max_w:
            break
    else:
        chosen = ["id"]
    for n in chosen:
        gi = fns[n](gi); go = fns[n](go)
    return {"input": np.asarray(gi).tolist(), "output": np.asarray(go).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = canvas colour (strict majority: object width < half grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    twos = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 2]
    two_rows = set(r for r, c in twos)
    two_cols = set(c for r, c in twos)
    vertical_bar = (len(two_cols) == 1)

    obj = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] not in (bgc, 2)]
    objc = int(I[obj[0]])
    r0 = min(r for r, c in obj); r1 = max(r for r, c in obj)
    c0 = min(c for r, c in obj); c1 = max(c for r, c in obj)

    if vertical_bar:
        b = next(iter(two_cols))
        ow = c1 - c0 + 1
        if c0 > b:
            steps = c0 - (b + 1)
            move_op, dstep = 23, (0, -1)
            line_idx = b + ow + 1
        else:
            steps = (b - 1) - c1
            move_op, dstep = 22, (0, 1)
            line_idx = b - ow - 1
        line_cells = [(r, line_idx) for r in range(hi) if 0 <= line_idx < wi]
    else:
        b = next(iter(two_rows))
        oh = r1 - r0 + 1
        if r0 > b:
            steps = r0 - (b + 1)
            move_op, dstep = 20, (-1, 0)
            line_idx = b + oh + 1
        else:
            steps = (b - 1) - r1
            move_op, dstep = 21, (1, 0)
            line_idx = b - oh - 1
        line_cells = [(line_idx, c) for c in range(wi) if 0 <= line_idx < hi]

    if steps > 0:
        dest = [(r + dstep[0] * steps, c + dstep[1] * steps) for r, c in obj]
        hole = sorted(set(obj) - set(dest))
        if objc != 0:
            # slide the object until it touches the 2-line: one grab, then empties
            ops.append(move_op); sels.append(sel_of(obj))
            for _ in range(steps - 1):
                ops.append(move_op); sels.append(sel_of([]))
            # repair only the vacated footprint (ARCLE left it at 0)
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # colour-0 object: Move cannot carry 0 cells; place it at its landing
            # spot with Color0, then clear the vacated footprint to background.
            ops.append(0); sels.append(sel_of(sorted(dest)))
            if hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))

    # draw the 8 frontier just beyond the object's far edge
    if line_cells:
        ops.append(8); sels.append(sel_of(line_cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
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
                                f"for task 56dc2b01"
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
                    f"Failed to build a complete episode for task 56dc2b01 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"56dc2b01-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
