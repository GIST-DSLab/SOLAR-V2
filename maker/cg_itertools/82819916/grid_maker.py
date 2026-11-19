"""
ARC Task: 82819916 (RE-ARC) — LLM-generated grid_maker
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
    # The rule only depends on the structure of the key line and the stubs, not on
    # which colours the lines use. Only the background has to be fixed.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    ass, bss = random.sample(remcols, 2)
    itv = list(range(w))
    while True:
        na = random.randint(2, w - 2)
        alocs = random.sample(itv, na)
        blocs = [j for j in itv if j not in alocs]
        if min(alocs) > min(blocs):
            alocs, blocs = blocs, alocs
        # Class B needs a column that survives truncation, so it can be read as a template.
        if min(blocs) < w - 1:
            break
    llocs = random.randint(0, h - 1)
    gi = [[bgc] * w for _ in range(h)]
    for j in alocs:
        gi[llocs][j] = ass
    for j in blocs:
        gi[llocs][j] = bss
    numl = unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = [r for r in range(h) if r != llocs]
    for _ in range(numl):
        lloc = random.choice(remlocs)
        remlocs.remove(lloc)
        a, b = random.sample(remcols, 2)
        for j in alocs:
            gi[lloc][j] = a
        for j in blocs:
            gi[lloc][j] = b
    cutoff = min(blocs) + 1
    go = [row[:] for row in gi]
    for r in range(h):
        if r == llocs:
            continue
        for j in range(cutoff, w):
            gi[r][j] = bgc
    gi = np.array(gi)
    go = np.array(go)
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Column-template reading: the key line sorts the cross-axis indices into two
    classes. Each class has at least one intact cross-section (a full column that
    still crosses every drawn line). That column is the template for its class.
    Every truncated column of the class becomes a copy of its template: CopyI the
    template column and Paste it onto each wiped column of the same class.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Background: the colour that fills the canvas behind the lines
    # (the majority colour, since lines occupy fewer than half the rows).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Orientation: the key line is the only full line with no background in it.
    horiz = any(not (I[r] == bgc).any() for r in range(hi))
    A = I if horiz else I.T
    h, w = A.shape

    key = next(r for r in range(h) if not (A[r] == bgc).any())
    kr = A[key]
    line_rows = [r for r in range(h) if (A[r] != bgc).any()]

    cA = int(kr[0])
    classA = [j for j in range(w) if int(kr[j]) == cA]
    classB = [j for j in range(w) if int(kr[j]) != cA]

    def intact(j):
        return all(int(A[r, j]) != bgc for r in line_rows)

    def col_sel(j):
        # A whole cross-section (full rectangle: one column, or one row if transposed)
        return [0, j, hi - 1, 0] if horiz else [j, 0, 0, wi - 1]

    def paste_sel(j):
        return [0, j, 0, 0] if horiz else [j, 0, 0, 0]

    def to_grid(r, j):
        return (r, j) if horiz else (j, r)

    ops, sels = [], []
    for cls in (classA, classB):
        templates = [j for j in cls if intact(j)]
        if not templates:
            continue
        t = templates[0]
        targets = [j for j in cls if not intact(j)]
        if not targets:
            continue
        # Copy the template cross-section from the input (the whole column, as a rectangle).
        ops.append(28); sels.append(col_sel(t))
        for j in targets:
            ops.append(30); sels.append(paste_sel(j))
        # Paste is transparent for 0. If the background is not 0, any 0 cells in the
        # template were not written. Paint them as Color0 on the pasted columns.
        if bgc != 0:
            zr = [r for r in line_rows if int(A[r, t]) == 0]
            if zr:
                cells = [to_grid(r, j) for j in targets for r in zr]
                ops.append(0); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 82819916"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 82819916"
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
                                f"for task 82819916"
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
                    f"Failed to build a complete episode for task 82819916 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"82819916-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
