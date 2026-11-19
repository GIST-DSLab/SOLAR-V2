"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, linc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])

    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    gi = canvas(bgc, (h, w))
    nl = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    inds = interval(0, h, 1)
    locs = []
    for k in range(nl):
        if len(inds) == 0:
            break
        idx = choice(inds)
        locs.append(idx)
        inds = remove(idx, inds)
        inds = remove(idx - 1, inds)
        inds = remove(idx + 1, inds)
        inds = remove(idx - 2, inds)
        inds = remove(idx + 2, inds)
    locs = sorted(locs)
    for ii in locs:
        gi = fill(gi, linc, hfrontier((ii, 0)))
    iopts = difference(
        difference(difference(interval(0, h, 1), locs), apply(increment, locs)),
        apply(decrement, locs),
    )
    jopts = interval(0, w, 1)
    ndots = unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for k in range(ndots):
        if len(iopts) == 0 or len(jopts) == 0:
            break
        loci = choice(iopts)
        locj = choice(jopts)
        dlocs.append((loci, locj))
        jopts = remove(locj, jopts)
        jopts = remove(locj + 1, jopts)
        jopts = remove(locj - 1, jopts)
    go = gi
    for d in dlocs:
        loci, locj = d
        if loci < min(locs):
            go = fill(go, dotc, connect(d, (min(locs), locj)))
            go = fill(go, linc, neighbors((min(locs), locj)))
        elif loci > max(locs):
            go = fill(go, dotc, connect(d, (max(locs), locj)))
            go = fill(go, linc, neighbors((max(locs), locj)))
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            go = fill(go, dotc, connect((sp, locj), (ep, locj)))
            go = fill(go, linc, neighbors((sp, locj)))
            go = fill(go, linc, neighbors((ep, locj)))
        gi = fill(gi, dotc, {d})
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured entirely from I):
      * The grid holds full straight lines of one colour (linc) plus isolated dots (dotc)
        on a background (bgc).  Lines are either all horizontal rows or (dmirrored task)
        all vertical columns.
      * Every dot shoots a ray of its own colour along the direction perpendicular to the
        lines, until it reaches the nearest line on each relevant side:
          - dot before the first line  -> ray to the first line
          - dot after  the last  line  -> ray to the last  line
          - dot between two lines      -> ray spanning from the previous to the next line
      * At every point where a ray meets a line, the 8 cells surrounding that crossing are
        painted with the line colour (the crossing cell itself keeps the dot colour).
    O is never inspected.
    """
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    ho, wo = Oa.shape
    hi, wi = A.shape

    # --- background: overwhelmingly the majority colour (lines <= h//4 rows, dots <= w//2)
    cnt = Counter(A.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # --- find the full lines and their orientation
    row_lines = [r for r in range(hi)
                 if len(set(A[r, :].tolist())) == 1 and int(A[r, 0]) != bgc]
    if row_lines:
        transposed = False
        G = A
        locs = row_lines
    else:
        transposed = True
        G = A.T
        locs = [c for c in range(wi)
                if len(set(A[:, c].tolist())) == 1 and int(A[0, c]) != bgc]

    H, W = G.shape
    linc = int(G[locs[0], 0])
    rest = sorted(c for c in cnt.keys() if c != bgc and c != linc)
    dotc = rest[0] if rest else linc

    def M(r, c):
        return (c, r) if transposed else (r, c)

    dots = sorted((r, c) for r in range(H) for c in range(W) if int(G[r, c]) == dotc)
    first_line, last_line = min(locs), max(locs)

    ops, sels = [], []
    for (r, c) in dots:
        if r < first_line:
            rows = list(range(r, first_line + 1))
            centers = [first_line]
        elif r > last_line:
            rows = list(range(last_line, r + 1))
            centers = [last_line]
        else:
            sp = max(x for x in locs if x < r)
            ep = min(x for x in locs if x > r)
            rows = list(range(sp, ep + 1))
            centers = [sp, ep]

        # the ray: paint the whole segment in the dot colour
        trail = [M(rr, c) for rr in rows]
        ops.append(int(dotc))
        sels.append(sel_of(trail))

        # each crossing point gets its 3x3 halo of line colour (centre stays the dot colour)
        for t in centers:
            ring = [M(rr, cc)
                    for rr in (t - 1, t, t + 1)
                    for cc in (c - 1, c, c + 1)
                    if 0 <= rr < H and 0 <= cc < W and not (rr == t and cc == c)]
            ops.append(int(linc))
            sels.append(sel_of(ring))

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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
