"""
ARC Task: 7f4411dc (RE-ARC) — LLM-generated grid_maker
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
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    if bgc is None or fgc is None:
        bgc, fgc = random.sample(range(10), 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    while tr < maxtr and succ < nsq:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        cands = [p for p in inds if p[0] <= h - oh and p[1] <= w - ow]
        if not cands:
            break
        loci, locj = random.choice(sorted(cands))
        obj = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        if obj <= inds:
            for (r, c) in obj:
                go[r][c] = fgc
            succ += 1
            outb = {(loci + a, locj + b) for a in range(-1, oh + 1) for b in range(-1, ow + 1)}
            inds = inds - outb
        tr += 1
    inds = {(r, c) for r in range(h) for c in range(w) if go[r][c] == bgc}
    ub = len(inds) // 2 - 1
    # at least one noise pixel so the demonstrations reveal which colour is background
    nnoise = unifint(diff_lb, diff_ub, (min(1, max(ub, 0)), max(ub, 0)))
    gi = [row[:] for row in go]
    for _ in range(nnoise):
        if not inds:
            break
        r, c = random.choice(sorted(inds))
        inds = inds - {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
        gi[r][c] = fgc
    return {"input": [list(row) for row in gi], "output": [list(row) for row in go]}


def _bgc_from(I, examples):
    I = np.asarray(I, dtype=int)
    # Convention read from the demonstrations: changed cells always turn INTO background.
    if examples:
        for ex in examples:
            a, b = (ex["input"], ex["output"]) if isinstance(ex, dict) else (ex[0], ex[1])
            a = np.asarray(a, dtype=int)
            b = np.asarray(b, dtype=int)
            if a.shape == b.shape:
                d = np.argwhere(a != b)
                if len(d):
                    return int(b[d[0][0], d[0][1]])
    # Fallback from I alone: true foreground's 1-thick cells are never adjacent to each other.
    hi, wi = I.shape
    best = None
    for col in sorted(set(I.flatten().tolist())):
        m = I != col
        thin = set()
        for r in range(hi):
            for c in range(wi):
                if m[r, c]:
                    hl = c > 0 and m[r, c - 1]
                    hr = c < wi - 1 and m[r, c + 1]
                    vu = r > 0 and m[r - 1, c]
                    vd = r < hi - 1 and m[r + 1, c]
                    if not (hl or hr) or not (vu or vd):
                        thin.add((r, c))
        clash = sum(1 for (r, c) in thin for d in ((0, 1), (1, 0))
                    if (r + d[0], c + d[1]) in thin)
        key = (clash, len(thin), -int((I == col).sum()))
        if best is None or key < best[0]:
            best = (key, col)
    return best[1]


def derive_operations(I, O, examples=None):
    # Reading (as in the golf entry): rectangles are >= 2 thick in both directions;
    # noise is anything only ONE cell thick along a line.  Sweep the lines:
    #   1) each row: foreground cells with no foreground left/right neighbour -> background
    #   2) each column (on the swept grid): foreground cells with no foreground
    #      above/below -> background (noise that touched a rectangle's side).
    # One Color op per row / column, selecting exactly that line's thin cells.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    cols = set(I.flatten().tolist())
    if len(cols) == 2:
        bgc = _bgc_from(I, examples)
        if bgc in cols:
            fgc = (cols - {bgc}).pop()
            G = I.copy()
            for r in range(hi):
                row = [c for c in range(wi) if G[r, c] == fgc
                       and not (c > 0 and G[r, c - 1] == fgc)
                       and not (c < wi - 1 and G[r, c + 1] == fgc)]
                if row:
                    ops.append(int(bgc)); sels.append(sel_of([(r, c) for c in row]))
                    for c in row:
                        G[r, c] = bgc
            for c in range(wi):
                colc = [r for r in range(hi) if G[r, c] == fgc
                        and not (r > 0 and G[r - 1, c] == fgc)
                        and not (r < hi - 1 and G[r + 1, c] == fgc)]
                if colc:
                    ops.append(int(bgc)); sels.append(sel_of([(r, c) for r in colc]))
                    for r in colc:
                        G[r, c] = bgc
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid, submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
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
                                f"for task 7f4411dc"
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
                    f"Failed to build a complete episode for task 7f4411dc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7f4411dc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
