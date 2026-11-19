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

# The golf entry and the maker read this task the same way, so the operations
# below follow the maker's reading.
#   Golf: out[r][i] = r[key.index(key[i])]. Each line is a lookup table from key
#   colour to line colour. The table is read at the first column carrying each key
#   colour, then spread over every column with that key colour.
#   Maker: the key line splits the columns into class A and class B. Each truncated
#   line gives one seed colour per class, and that colour fills the whole class.
# Both readings use the same unit (one line, one key-colour class at a time) and
# keep the same thing fixed (the key line's partition). So each line gets one Color
# op per class: "all columns where the key shows colour X, on this line".

ROTS = [0, 1, 2, 3]  # identity, rot90, rot180, rot270


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": k} for k in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        # guarantee both orientations (row key / column key) when possible
        picks = random.sample(ROTS, n_ex)
        if n_ex >= 2 and len({k % 2 for k in picks}) == 1:
            picks[-1] = random.choice([k for k in ROTS if k % 2 != picks[0] % 2])
        ex = [{"rot": k} for k in picks]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    if rot is None:
        rot = random.choice(ROTS)
    cols = list(range(10))
    # after rotation by 90/270 the shape swaps, so bound accordingly
    if rot % 2 == 0:
        H, W = max_h, max_w
    else:
        H, W = max_w, max_h
    h = unifint(diff_lb, diff_ub, (5, min(30, H)))
    w = unifint(diff_lb, diff_ub, (5, min(30, W)))
    remcols = [c for c in cols if c != bgc]
    ass, bss = random.sample(remcols, 2)
    itv = list(range(w))
    na = random.randint(2, w - 2)
    alocs = random.sample(itv, na)
    blocs = [j for j in itv if j not in alocs]
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = random.randint(0, h - 1)
    gi = np.full((h, w), bgc, dtype=int)
    for j in alocs:
        gi[llocs, j] = ass
    for j in blocs:
        gi[llocs, j] = bss
    numl = unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = [r for r in range(h) if r != llocs]
    for _ in range(numl):
        lloc = random.choice(remlocs)
        remlocs.remove(lloc)
        a, b = random.sample(remcols, 2)
        for j in alocs:
            gi[lloc, j] = a
        for j in blocs:
            gi[lloc, j] = b
    cutoff = min(blocs) + 1
    go = gi.copy()
    gi[:, cutoff:] = bgc
    for j in alocs:
        gi[llocs, j] = ass
    for j in blocs:
        gi[llocs, j] = bss
    # rot90 (CW) = np.rot90(k=3); rot270 = np.rot90(k=1)
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background: the colour every demonstration input shares as its majority
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # orientation, measured from I: is the full (bg-free) key line a row or a column?
    horiz = any(not (I[r] == bgc).any() for r in range(hi))
    A = I if horiz else I.T
    h, w = A.shape

    def to_grid(r, j):
        return (r, j) if horiz else (j, r)

    key = next(r for r in range(h) if not (A[r] == bgc).any())
    kr = A[key]
    key_colours = []
    for j in range(w):
        if int(kr[j]) not in key_colours:
            key_colours.append(int(kr[j]))
    # one class per key colour: the columns where the key line shows that colour
    classes = {kc: [j for j in range(w) if int(kr[j]) == kc] for kc in key_colours}

    ops, sels = [], []
    cur = A.copy()
    for r in range(h):
        if r == key:
            continue
        if not (A[r] != bgc).any():
            continue
        for kc in key_colours:
            cls = classes[kc]
            # lookup: this line's colour at the first drawn column of this class
            seeds = [j for j in cls if int(A[r, j]) != bgc]
            if not seeds:
                continue
            col = int(A[r, seeds[0]])
            # paint the whole class on this line; skip only cells already holding col
            cells_j = [j for j in cls if int(cur[r, j]) != col]
            if not cells_j:
                continue
            for j in cells_j:
                cur[r, j] = col
            ops.append(col)
            sels.append(sel_of([to_grid(r, j) for j in cells_j]))

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
