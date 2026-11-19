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
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))
    if bgc is None:
        bgc = random.choice(range(10))
    h = unif(5, min(30, max_h))
    w = unif(5, min(30, max_w))
    remcols = [c for c in range(10) if c != bgc]
    ass, bss = random.sample(remcols, 2)
    itv = list(range(w))
    na = random.randint(2, w - 2)
    alocs = random.sample(itv, na)
    blocs = [j for j in itv if j not in alocs]
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = random.randint(0, h - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[llocs, alocs] = ass
    gi[llocs, blocs] = bss
    numl = unif(1, max(1, (h - 1) // 2))
    remlocs = [r for r in range(h) if r != llocs]
    for _ in range(numl):
        lloc = random.choice(remlocs)
        remlocs.remove(lloc)
        a, b = random.sample(remcols, 2)
        gi[lloc, alocs] = a
        gi[lloc, blocs] = b
    cutoff = min(blocs) + 1
    go = gi.copy()
    gi[:, cutoff:] = bgc
    gi[llocs, alocs] = ass
    gi[llocs, blocs] = bss
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (column-as-unit, as in the golf entry): the one complete line (key line)
    is bicoloured and splits the perpendicular lines into class A / class B.
    The surviving stub contains at least one COMPLETE perpendicular line of each
    class; every other perpendicular line of the same class is a copy of it.
    So: CopyI a complete representative line of class A, Paste it onto every
    incomplete class-A line; same for class B.  Paste is transparent to 0, so if
    the representative carries real 0 cells (bgc != 0), those cells are painted
    with Color0 afterwards across the pasted lines.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    horiz = any(not (I[r] == bgc).any() for r in range(hi))
    A = I if horiz else I.T
    h, w = A.shape

    def g(r, j):
        return (r, j) if horiz else (j, r)

    def line_sel(j):  # whole perpendicular line j (full rectangle: one column/row)
        return [0, j, h - 1, 0] if horiz else [j, 0, 0, h - 1]

    key = next(r for r in range(h) if not (A[r] == bgc).any())
    kr = A[key]
    cA = int(kr[0])
    drawn = [r for r in range(h) if (A[r] != bgc).any()]
    intact = [j for j in range(w) if all(A[r, j] != bgc for r in drawn)]

    grid = I.copy()
    G = grid if horiz else grid.T  # view in canonical orientation
    ops, sels = [], []
    for is_a in (True, False):
        cls = [j for j in range(w) if (int(kr[j]) == cA) == is_a]
        rep = next(j for j in cls if j in intact)
        targets = [j for j in cls if j not in intact]
        if not targets:
            continue
        repcol = A[:, rep].copy()
        copied = False
        for j in targets:
            new = G[:, j].copy()
            nz = repcol != 0
            new[nz] = repcol[nz]
            if (new == G[:, j]).all():
                continue
            if not copied:
                ops.append(28); sels.append(line_sel(rep))
                copied = True
            ops.append(30); sels.append(sel_of([g(0, j)]))
            G[:, j] = new
        zr = [r for r in range(h) if repcol[r] == 0]
        cells = [g(r, j) for j in targets for r in zr if G[r, j] != 0]
        if cells:
            ops.append(0); sels.append(sel_of(cells))
            for j in targets:
                for r in zr:
                    G[r, j] = 0
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
