"""
ARC Task: a68b268e (RE-ARC) — LLM-generated grid_maker
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


def _unif(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * max(0.0, min(1.0, diff_lb))
    b = lo + (hi - lo) * max(0.0, min(1.0, diff_ub))
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    return random.randint(max(lo, a), min(hi, max(a, b)))


def sample_colors(num_examples=None) -> dict:
    bgc, linc, c1, c2, c3, c4 = random.sample(list(range(10)), 6)
    return {"bgc": bgc, "linc": linc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, linc=1, c1=2, c2=3, c3=4, c4=5, **kw) -> dict:
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(4, (max_w - 1) // 2))
    h = _unif(diff_lb, diff_ub, 2, hmax)
    w = _unif(diff_lb, diff_ub, 2, wmax)
    inds = [(i, j) for i in range(h) for j in range(w)]

    def ncount():
        d = _unif(diff_lb, diff_ub, 0, (h * w) // 2)
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)

    ofs = [random.sample(inds, ncount()) for _ in range(4)]
    cols = [c1, c2, c3, c4]
    go = [[bgc] * w for _ in range(h)]
    for col, of in zip(cols, ofs):
        for (i, j) in of:
            go[i][j] = col

    def layer(col, of):
        g = [[bgc] * w for _ in range(h)]
        for (i, j) in of:
            g[i][j] = col
        return g

    LR = layer(c1, ofs[0]); LL = layer(c2, ofs[1])
    UR = layer(c3, ofs[2]); UL = layer(c4, ofs[3])
    gi = [[linc] * (2 * w + 1) for _ in range(2 * h + 1)]
    for (g, dr, dc) in ((LR, h + 1, w + 1), (LL, h + 1, 0), (UR, 0, w + 1), (UL, 0, 0)):
        for i in range(h):
            for j in range(w):
                gi[i + dr][j + dc] = g[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2

    quads = [I[0:h, 0:w], I[0:h, wi - w:wi], I[hi - h:hi, 0:w], I[hi - h:hi, wi - w:wi]]
    common = set(quads[0].flatten().tolist())
    for q in quads[1:]:
        common &= set(q.flatten().tolist())
    bgc = min(common) if common else Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    G = I.copy()

    # Fold reading: first fold each half left<-right (right fills left's background),
    # then fold top<-bottom (bottom-combined fills top-combined's background).
    top_comb = np.where(quads[0] != bgc, quads[0], quads[1])
    # cells of the top half still background after stage 1 (determined from input)
    top_holes = top_comb == bgc

    def fold(dst_r, src_r, src_c, relevant=None):
        groups = {}
        for r in range(h):
            for c in range(w):
                if G[dst_r + r, c] != bgc:
                    continue
                v = G[src_r + r, src_c + c]
                if v == bgc:
                    continue
                groups.setdefault(int(v), []).append((dst_r + r, c, r))
        for col in sorted(groups):
            cells = groups[col]
            if relevant is not None and not any(relevant[r, c] for (_, c, r) in cells):
                continue  # this fill can never show in the final quadrant
            sel = [(rr, cc) for (rr, cc, _) in cells]
            for (rr, cc) in sel:
                G[rr, cc] = col
            ops.append(col)
            sels.append(sel_of(sel))

    # stage 1: top half — UR fills UL's background
    fold(0, 0, wi - w)
    # stage 1: bottom half — LR fills LL's background
    fold(hi - h, hi - h, wi - w, relevant=top_holes)
    # stage 2: bottom-combined (LL region) fills top-combined (UL region)
    fold(0, hi - h, 0)

    # crop to the assembled upper-left quadrant (full rectangle)
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a68b268e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a68b268e"
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
                                f"for task a68b268e"
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
                    f"Failed to build a complete episode for task a68b268e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a68b268e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
