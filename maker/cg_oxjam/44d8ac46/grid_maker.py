"""
ARC Task: 44d8ac46 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Fill colour 2 is hardcoded by the rule; only the background is sampled (never 2).
    cols = [c for c in range(10) if c != 2]
    return {"bgc": random.choice(cols)}


def _unifint(diff_lb, diff_ub, lb, ub):
    d = random.uniform(diff_lb, diff_ub)
    return max(lb, min(ub, lb + int(round(d * (ub - lb)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, 10, max(10, max_h))
    w = _unifint(diff_lb, diff_ub, 10, max(10, max_w))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = _unifint(diff_lb, diff_ub, 1, 10)
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = succ = 0
    while succ < num and tr <= maxtrials:
        tr += 1
        if not indss or not remcols:
            break
        oh = random.randint(5, 7)
        ow = random.randint(5, 7)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            continue
        loci, locj = random.choice(sorted(subs))
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        col = random.choice(remcols)
        if not bd <= indss:
            continue
        if random.choice((True, False)):
            dim = random.randint(1, min(oh, ow) - 2)
            il = random.randint(1, oh - dim - 1)
            jl = random.randint(1, ow - dim - 1)
            inpart = {(loci + il + a, locj + jl + b) for a in range(dim) for b in range(dim)}
        else:
            cnds = {(i, j) for i in range(loci + 1, loci + oh - 1)
                    for j in range(locj + 1, locj + ow - 1)}
            inpart = {random.choice(sorted(cnds))}
            kk = _unifint(diff_lb, diff_ub, 1, len(cnds))
            for _ in range(kk - 1):
                nb = {(i + di, j + dj) for i, j in inpart
                      for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}
                cand = sorted((cnds - inpart) & nb)
                if not cand:
                    break
                inpart.add(random.choice(cand))
        rs = [p[0] for p in inpart]
        cs = [p[1] for p in inpart]
        ih = max(rs) - min(rs) + 1
        iw = max(cs) - min(cs) + 1
        incol = 2 if (ih == iw and len(inpart) == ih * iw) else bgc
        for i, j in bd:
            gi[i][j] = col
            go[i][j] = col
        for i, j in inpart:
            gi[i][j] = bgc
            go[i][j] = incol
        succ += 1
        ob = {(i, j) for i in range(loci - 1, loci + oh + 1)
              for j in range(locj - 1, locj + ow + 1)}
        indss = indss - ob
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import deque
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Boxes never touch the last row/col, so that corner is background.
    bgc = int(I[hi - 1, wi - 1])

    ops, sels = [], []
    # Unit of the rule = an ENCLOSED background pocket: a 4-connected background
    # region that never reaches the grid border (the hollow inside one frame).
    # A pocket that is a solid square is flooded with 2 from a single seed cell;
    # FloodFill spreads over exactly that pocket and nothing else.
    seen = np.zeros((hi, wi), dtype=bool)
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] != bgc:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            border = False
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                if y in (0, hi - 1) or x in (0, wi - 1):
                    border = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == bgc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if border:
                continue  # the open outside background, not a hole
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            hh = max(ys) - min(ys) + 1
            ww = max(xs) - min(xs) + 1
            if hh == ww and len(cells) == hh * ww:
                ops.append(12)  # FloodFill2 seeded inside the square pocket
                sels.append(sel_of([min(cells)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
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
                                f"for task 44d8ac46"
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
                    f"Failed to build a complete episode for task 44d8ac46 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44d8ac46-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
