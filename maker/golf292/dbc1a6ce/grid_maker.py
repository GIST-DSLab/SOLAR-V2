"""
ARC Task: dbc1a6ce (RE-ARC) — LLM-generated grid_maker
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
    colopts = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    bgc = random.choice(colopts)
    fgcol = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgcol": fgcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgcol=None) -> dict:
    colopts = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    if bgc is None:
        bgc = random.choice(colopts)
    if fgcol is None or fgcol == bgc:
        fgcol = random.choice([c for c in colopts if c != bgc])

    def unif(lb, ub):
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, int(round(lb + d * (ub - lb)))))

    h = unif(3, max(3, max_h))
    w = unif(3, max(3, max_w))
    inds = [(r, c) for r in range(h) for c in range(w)]
    num = unif(0, max(1, (h * w) // 4))
    s = random.sample(inds, num)
    gi = [[bgc] * w for _ in range(h)]
    for r, c in s:
        gi[r][c] = fgcol
    go = [[bgc] * w for _ in range(h)]
    for r in range(h):
        m = [c for c in range(w) if gi[r][c] == fgcol]
        if len(m) > 1:
            for c in range(m[0], m[-1] + 1):
                go[r][c] = 8
    for c in range(w):
        m = [r for r in range(h) if gi[r][c] == fgcol]
        if len(m) > 1:
            for r in range(m[0], m[-1] + 1):
                go[r][c] = 8
    for r, c in s:
        go[r][c] = fgcol
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Rule (read from I alone): fgc = least frequent colour.  Every row / column
    holding >=2 fgc marks gets a line of 8 spanning first..last mark; the marks
    themselves stay fgc.
    Ops: one Color8 stroke per row/column span (whole segment, marks included),
    then one Color(fgc) on all fgc cells of I to restore the marks.
    A span whose non-mark cells are all covered by other kept spans is dropped
    (it would be redundant).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    cnt = Counter(I.flatten().tolist())
    if len(cnt) >= 2:
        fgc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        spans = []
        for r in range(hi):
            m = [c for c in range(wi) if I[r, c] == fgc]
            if len(m) >= 2:
                spans.append([(r, c) for c in range(m[0], m[-1] + 1)])
        for c in range(wi):
            m = [r for r in range(hi) if I[r, c] == fgc]
            if len(m) >= 2:
                spans.append([(r, c) for r in range(m[0], m[-1] + 1)])
        keep = list(range(len(spans)))
        for i in range(len(spans)):
            own = {p for p in spans[i] if I[p] != fgc}
            others = set()
            for j in keep:
                if j != i:
                    others.update(spans[j])
            if not (own - others):
                keep.remove(i)
        for i in keep:
            cells = spans[i]
            (r0, c0), (r1, c1) = cells[0], cells[-1]
            # full 1-wide rectangle: the row/column segment between first and last mark
            ops.append(8)
            sels.append([r0, c0, r1 - r0, c1 - c0])
        if keep:
            marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]
            ops.append(int(fgc))
            sels.append(sel_of(marks))
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
                        f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
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
                                f"for task dbc1a6ce"
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
                    f"Failed to build a complete episode for task dbc1a6ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dbc1a6ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
