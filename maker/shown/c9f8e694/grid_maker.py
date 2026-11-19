"""
ARC Task: c9f8e694 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback for standalone use
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ---------------------------------------------------------------- 1. colors --

# The only structural degree of freedom is the final rotation, which decides
# WHERE the key-line lives (left column / top row / right column / bottom row).
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # sqc (the colour of the rectangles) is sampled by the generator -> fix it
    # for the whole episode.  bgc is hardcoded to 0 in the generator, and the
    # key-line colours are pure content (the rule copies whatever is there).
    sqc = random.choice(list(range(1, 10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"sqc": sqc, "instance_plan": plan}


# -------------------------------------------------------------- 2. generate --

def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             sqc=None, rot=None) -> dict:
    cols = interval(1, 10, 1)
    if sqc is None:
        sqc = random.choice(cols)
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    # a 90/270 rotation swaps the axes, so bound the pre-rotation dims accordingly
    hb = max_h if rot % 2 == 0 else max_w
    wb = max_w if rot % 2 == 0 else max_h
    hb = max(5, min(30, int(hb)))
    wb = max(5, min(30, int(wb)))

    bgc = 0
    remcols = remove(bgc, cols)
    remcols = remove(sqc, remcols)

    while True:
        h = unifint(diff_lb, diff_ub, (5, hb))
        w = unifint(diff_lb, diff_ub, (5, wb))
        nsq = unifint(diff_lb, diff_ub, (1, 8))
        gir = canvas(bgc, (h, w - 1))
        gil = tuple((random.choice(remcols),) for j in range(h))
        inds = asindices(gir)
        succ = 0
        fails = 0
        maxfails = nsq * 5
        while succ < nsq and fails < maxfails:
            loci = random.randint(0, h - 3)
            locj = random.randint(0, w - 3)
            lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
            locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
            bd = backdrop(frozenset({(loci, locj), (lock, locl)}))
            if bd.issubset(inds):
                gir = fill(gir, sqc, bd)
                succ += 1
            else:
                fails += 1
        if succ > 0:
            break

    locs = ofcolor(gir, sqc)
    gil = tuple(e if idx in apply(first, locs) else (bgc,) for idx, e in enumerate(gil))
    fullobj = toobject(locs, hupscale(gil, w))
    gi = hconcat(gil, gir)
    giro = paint(gir, fullobj)
    go = hconcat(gil, giro)
    rotf = (identity, rot90, rot180, rot270)[rot]
    return {'input': rotf(gi), 'output': rotf(go)}


# ------------------------------------------------------------ 3. operations --

def _sqc_from_examples(examples):
    """The block colour is fixed for the episode and never survives into an
    output (the key colours are drawn from a palette that excludes it), so it is
    the colour present in every demo input and absent from every demo output."""
    if not examples:
        return None
    cand = set(range(1, 10))
    for pair in examples:
        try:
            ei = np.asarray(pair[0], dtype=int)
            eo = np.asarray(pair[1], dtype=int)
        except Exception:
            return None
        cand &= (set(np.unique(ei).tolist()) - set(np.unique(eo).tolist()))
        if not cand:
            return None
    if len(cand) == 1:
        return cand.pop()
    return None


def _check_line(I, kind, idx, sqc_hint):
    """Is this border line the key-line?  `rest[i]` is the perpendicular
    line belonging to key entry i."""
    if kind == 'col':
        key = I[:, idx]
        rest = np.delete(I, idx, axis=1)
    else:
        key = I[idx, :]
        rest = np.delete(I, idx, axis=0).T

    vals = set(np.unique(rest).tolist()) - {0}
    if len(vals) != 1:
        return None
    sqc = vals.pop()
    if sqc_hint is not None and sqc != sqc_hint:
        return None
    if not key.any():
        return None
    if sqc in set(key.tolist()):
        return None
    # a key entry is coloured exactly when its perpendicular line carries blocks
    for i in range(len(key)):
        if bool(key[i] != 0) != bool((rest[i] == sqc).any()):
            return None
    return sqc, np.array(key, dtype=int)


def _find_key(I, sqc_hint):
    h, w = I.shape
    cands = [('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)]
    for hint in (sqc_hint, None):
        for kind, idx in cands:
            res = _check_line(I, kind, idx, hint)
            if res is not None:
                sqc, key = res
                return kind, idx, key, sqc
    return None


def derive_operations(I, O, examples=None):
    """Every block cell takes the colour of the key-line entry sitting on its
    own row (key = a column) or its own column (key = a row).  Everything below
    is measured from I and from the demonstrations; O is never inspected."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    ops, sels = [], []

    sqc_hint = _sqc_from_examples(examples)
    found = _find_key(I, sqc_hint)

    if found is not None:
        kind, key_idx, key, sqc = found
        # walk along the key-line, recolouring one stripe of blocks per entry
        for i in range(len(key)):
            col = int(key[i])
            if col == 0:
                continue
            if kind == 'col':
                cells = [(i, c) for c in range(w) if c != key_idx and I[i, c] == sqc]
            else:
                cells = [(r, i) for r in range(h) if r != key_idx and I[r, i] == sqc]
            if not cells:
                continue
            ops.append(col)            # Color<key colour> on that stripe's block cells
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
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
                                f"for task c9f8e694"
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
                    f"Failed to build a complete episode for task c9f8e694 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c9f8e694-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
