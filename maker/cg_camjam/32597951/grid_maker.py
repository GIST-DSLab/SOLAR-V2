"""
ARC Task: 32597951 (RE-ARC) — LLM-generated grid_maker
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
    # 3 is hardcoded as the output colour, so it is excluded from every role.
    cols = [c for c in range(10) if c != 3]
    bgc, noisec, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, fgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if bgc is None or noisec is None or fgc is None:
        cols = [c for c in range(10) if c != 3]
        bgc, noisec, fgc = random.sample(cols, 3)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    c = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    ndev = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    num = random.choice((ndev, h * w - ndev))
    num = min(max(num, 0), h * w)
    for (i, j) in random.sample(inds, num):
        c[i, j] = noisec
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    tofill = [(i, j) for i in range(loci, loci + ih) for j in range(locj, locj + iw)
              if c[i, j] == bgc]
    gi = c.copy()
    for (i, j) in tofill:
        gi[i, j] = fgc
    go = gi.copy()
    if tofill:
        rs = [p[0] for p in tofill]
        cs = [p[1] for p in tofill]
        for i in range(min(rs), max(rs) + 1):
            for j in range(min(cs), max(cs) + 1):
                if gi[i, j] == noisec:
                    go[i, j] = 3
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Golf reading: one marker colour (fixed per episode) marks the block. The block
    # is every row that holds the marker crossed with every column that holds it.
    # Inside that block, any cell that is not the marker turns 3. This matches the
    # maker's bbox-minus-object rule, so the ops come out the same. The only change
    # is how the marker colour is found: we read it from the demonstrations, not
    # from "smallest bbox".
    fgc = None
    if examples:
        cand = None
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            eo = np.asarray(ex[1], dtype=int)
            threes = list(zip(*np.where(eo == 3)))
            if not threes:
                continue
            ok = set()
            for col in np.unique(ei):
                rr = set(np.where((ei == col).any(axis=1))[0].tolist())
                cc = set(np.where((ei == col).any(axis=0))[0].tolist())
                if all(r in rr and c in cc for r, c in threes):
                    ok.add(int(col))
            cand = ok if cand is None else (cand & ok)
        if cand:
            present = [c for c in cand if (I == c).any()]
            if len(present) == 1:
                fgc = present[0]
            elif present:
                # break ties by the most compact rows x cols block in I
                def area(col):
                    rr = (I == col).any(axis=1).sum()
                    cc = (I == col).any(axis=0).sum()
                    return rr * cc
                fgc = min(present, key=area)
    if fgc is None:
        best = None
        for col in np.unique(I):
            rs, cs = np.where(I == col)
            a = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
            if best is None or a < best[0]:
                best = (a, int(col))
        fgc = best[1]

    ops, sels = [], []
    if (I == fgc).any():
        rows = np.where((I == fgc).any(axis=1))[0]
        cols = np.where((I == fgc).any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        # The non-marker cells in the marker's rows x columns block.
        targets = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if I[r, c] != fgc]
        if targets:
            ops.append(3)
            sels.append(sel_of(targets))

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
                        f"num_examples+1 ({num_examples + 1}) for task 32597951"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 32597951"
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
                                f"for task 32597951"
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
                    f"Failed to build a complete episode for task 32597951 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"32597951-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
