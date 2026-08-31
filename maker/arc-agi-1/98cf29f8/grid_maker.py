"""
ARC Task: 98cf29f8 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    # objc / otherc must be non-zero: the mover object is relocated with ARCLE Move ops,
    # and Move only carries NON-ZERO cells of the selection.
    rest = [c for c in cols if c != bgc and c != 0]
    objc, otherc = random.sample(rest, 2)

    # Discrete structural variant: which way the stemmed rectangle travels.
    variants = [{"direction": d} for d in ("up", "down", "left", "right")]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "otherc": otherc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, otherc, direction=None) -> dict:
    if direction is None:
        direction = random.choice(("up", "down", "left", "right"))

    def ui(bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))

    # any of the 4 orientations may transpose h/w, so bound both by the smaller limit
    m = min(max_h, max_w)
    if m < 10:
        m = 10

    h = ui((10, m))
    w = ui((10, m))
    objh = ui((2, h - 5))
    objw = ui((2, w - 5))
    # anchor block placed with a bottom margin > 2 (base orientation: mover travels UP)
    loci = random.randint(0, h - objh - 3)
    locj = random.randint(0, w - objw)

    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + objh):
        for c in range(locj, locj + objw):
            gi[r][c] = objc

    low = loci + objh - 1
    left = locj
    right = locj + objw - 1

    locis = random.randint(low + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, right))
    locje = random.randint(max(locjs + 1, left), w - 1)
    jloc = random.randint(max(left, locjs), min(right, locje))
    lnlen = locis - low - 1

    go = [row[:] for row in gi]
    for r in range(locis, locie + 1):
        for c in range(locjs, locje + 1):
            gi[r][c] = otherc
            go[r - lnlen][c] = otherc
    for r in range(low + 1, locis):
        gi[r][jloc] = otherc

    def flipud(g):
        return [row[:] for row in g[::-1]]

    def transpose(g):
        return [list(t) for t in zip(*g)]

    def rot_cw(g):
        return [list(t) for t in zip(*g[::-1])]

    if direction == "down":
        gi, go = flipud(gi), flipud(go)
    elif direction == "left":
        gi, go = transpose(gi), transpose(go)
    elif direction == "right":
        gi, go = rot_cw(gi), rot_cw(go)

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    # background = most common colour on the border ring (objects can dominate the interior)
    border = ([int(I[0, c]) for c in range(w)] + [int(I[h - 1, c]) for c in range(w)] +
              [int(I[r, 0]) for r in range(h)] + [int(I[r, w - 1]) for r in range(h)])
    bg = Counter(border).most_common(1)[0][0]

    ops, sels = [], []

    # the mover is the colour whose cell set differs between I and O; the other block is the anchor
    M, Mo = set(), set()
    for col in sorted(set(I.flatten().tolist()) - {bg}):
        a = {(r, c) for r in range(h) for c in range(w) if int(I[r, c]) == col}
        b = {(r, c) for r in range(ho) for c in range(wo) if int(O[r, c]) == col}
        if a != b:
            M, Mo = a, b
            break

    # split the mover into its solid rectangle and its 1-wide connecting stem
    stem = []
    for (r, c) in M:
        l_in = (r, c - 1) in M
        r_in = (r, c + 1) in M
        u_in = (r - 1, c) in M
        d_in = (r + 1, c) in M
        if (not l_in and not r_in) or (not u_in and not d_in):
            stem.append((r, c))
    stem = sorted(stem)
    rect = sorted(M - set(stem))

    # destination of the rectangle (pure translation along the stem axis)
    r0 = min(r for r, c in rect)
    c0 = min(c for r, c in rect)
    dr = min(r for r, c in Mo) - r0
    dc = min(c for r, c in Mo) - c0
    dest = {(r + dr, c + dc) for (r, c) in rect}

    # erase the stem only when the arriving rectangle would not cover all of it
    if stem and any(p not in dest for p in stem):
        ops.append(int(bg))
        sels.append(sel_of(stem))

    # slide the rectangle until it abuts the anchor block
    steps = abs(dr) + abs(dc)
    if dr < 0:
        mop = 20
    elif dr > 0:
        mop = 21
    elif dc > 0:
        mop = 22
    else:
        mop = 23
    if steps > 0:
        ops.append(mop)
        sels.append(sel_of(rect))               # first Move GRABS the rectangle
        for _ in range(steps - 1):
            ops.append(mop)
            sels.append(sel_of([]))             # empty selection -> keep same object grabbed

    # ARCLE leaves the vacated original footprint at 0; repair it when bg != 0
    hole = sorted(set(rect) - dest)
    if bg != 0 and hole:
        ops.append(int(bg))
        sels.append(sel_of(hole))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
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
                                f"for task 98cf29f8"
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
                    f"Failed to build a complete episode for task 98cf29f8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"98cf29f8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
