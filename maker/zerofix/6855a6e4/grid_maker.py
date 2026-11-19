"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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

# discrete structural variant: the whole scene is transposed (brackets horizontal vs vertical)
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # bgc, objc (the dots that get reflected), boxc (the two brackets) are all sampled
    # randomly by the original generator -> fix all three for the episode.
    bgc, objc, boxc = random.sample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        exs = [dict(v) for v in VARIANTS]
        exs += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(exs)
    else:
        exs = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = exs + [dict(random.choice(exs))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transpose=None) -> dict:
    g = globals()

    def U(lo, hi):
        if hi <= lo:
            return lo
        if 'unifint' in g:
            return g['unifint'](diff_lb, diff_ub, (lo, hi))
        return random.randint(lo, hi)

    if transpose is None:
        transpose = random.choice([True, False])

    # after a transpose the sampled h becomes the final width
    mh = max_w if transpose else max_h
    mw = max_h if transpose else max_w

    h = U(10, max(10, mh))
    w = U(4, max(4, mw))
    fullh = U(10, max(10, h))
    fullw = U(3, max(3, w))
    h = max(h, fullh)
    w = max(w, fullw)
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)

    objh = (fullh // 2 - 3) // 2          # rows of the dot region on each side
    regi = [[bgc] * fullw for _ in range(fullh)]
    rego = [[bgc] * fullw for _ in range(fullh)]

    top_line = objh + 1
    bot_line = fullh - 2 - objh
    for reg in (regi, rego):
        for c in range(fullw):
            reg[top_line][c] = boxc
            reg[bot_line][c] = boxc
        reg[top_line + 1][0] = boxc
        reg[top_line + 1][fullw - 1] = boxc
        reg[bot_line - 1][0] = boxc
        reg[bot_line - 1][fullw - 1] = boxc

    ncols = fullw - 2
    total = objh * ncols
    cands = [(r, c) for r in range(objh) for c in range(1, fullw - 1)]
    first_row = [p for p in cands if p[0] == 0]
    last_row = [p for p in cands if p[0] == objh - 1]

    for side in range(2):
        nd = U(0, max(0, total // 2))
        ncells = random.choice([nd, total - nd])
        ncells = min(max(1, ncells), total)
        # force the dot group to touch both edges of its region, so the rule is exactly
        # "reflect the dots across the bracket line" and the episode stays learnable
        forced = {random.choice(first_row)}
        if objh > 1:
            forced.add(random.choice(last_row))
        rest = [p for p in cands if p not in forced]
        k = max(0, min(len(rest), ncells - len(forced)))
        cells = list(forced) + random.sample(rest, k)
        for (r, c) in cells:
            if side == 0:
                ri = r
                ro = 2 * top_line - r
            else:
                ri = fullh - 1 - r
                ro = 2 * bot_line - (fullh - 1 - r)
            regi[ri][c] = objc
            rego[ro][c] = objc

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(fullh):
        for c in range(fullw):
            gi[loci + r][locj + c] = regi[r][c]
            go[loci + r][locj + c] = rego[r][c]

    if transpose:
        gi = [list(row) for row in zip(*gi)]
        go = [list(row) for row in zip(*go)]

    return {'input': tuple(tuple(row) for row in gi),
            'output': tuple(tuple(row) for row in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    def bg_of(gr):
        gr = np.asarray(gr, dtype=int)
        return Counter(gr.flatten().tolist()).most_common(1)[0][0]

    cnt = Counter(I.flatten().tolist())
    bgc = bg_of(I)
    if examples:
        votes = Counter([bg_of(e[0]) for e in examples] + [bgc])
        best = votes.most_common(1)[0][0]
        if best in cnt:
            bgc = best

    cand = [c for c in cnt if c != bgc]

    def box_shape(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        S = set(cells)
        if r1 - r0 >= 3 and c1 - c0 >= 2:
            H = set()
            for c in range(c0, c1 + 1):
                H.add((r0, c)); H.add((r1, c))
            H |= {(r0 + 1, c0), (r0 + 1, c1), (r1 - 1, c0), (r1 - 1, c1)}
            if S == H:
                return ('h', r0, r1, c0, c1)
        if c1 - c0 >= 3 and r1 - r0 >= 2:
            V = set()
            for r in range(r0, r1 + 1):
                V.add((r, c0)); V.add((r, c1))
            V |= {(r0, c0 + 1), (r1, c0 + 1), (r0, c1 - 1), (r1, c1 - 1)}
            if S == V:
                return ('v', r0, r1, c0, c1)
        return None

    boxc, shape = None, None
    for c in cand:
        cells = [(r, cc) for r in range(h) for cc in range(w) if I[r, cc] == c]
        s = box_shape(cells)
        if s is not None:
            boxc, shape = c, s
    if shape is None:                       # defensive fallback: brackets span their bbox
        for c in cand:
            cells = [(r, cc) for r in range(h) for cc in range(w) if I[r, cc] == c]
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            S = set(cells)
            if all((r0, cc) in S for cc in range(c0, c1 + 1)):
                boxc, shape = c, ('h', r0, r1, c0, c1)
            elif all((rr, c0) in S for rr in range(r0, r1 + 1)):
                boxc, shape = c, ('v', r0, r1, c0, c1)
    objc = [c for c in cand if c != boxc][0]

    kind, r0, r1, c0, c1 = shape
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == objc]
    if kind == 'h':
        groups = [([p for p in dots if p[0] < r0], r0),
                  ([p for p in dots if p[0] > r1], r1)]
    else:
        groups = [([p for p in dots if p[1] < c0], c0),
                  ([p for p in dots if p[1] > c1], c1)]

    ops, sels = [], []
    for cells, B in groups:
        if not cells:
            continue
        if kind == 'h':
            u = min(r for r, _ in cells); l = max(r for r, _ in cells)
            flipped = [(u + l - r, c) for r, c in cells]
            d = 2 * B - u - l
            mv = 21 if d > 0 else 20          # slide toward / through the bracket
            flip_op = 27                      # FlipV = up<->down
            dst = [(r + d, c) for r, c in flipped]
        else:
            u = min(c for _, c in cells); l = max(c for _, c in cells)
            flipped = [(r, u + l - c) for r, c in cells]
            d = 2 * B - u - l
            mv = 22 if d > 0 else 23
            flip_op = 26                      # FlipH = left<->right
            dst = [(r, c + d) for r, c in flipped]

        if objc == 0:
            # 0 is transparent to ARCLE's object ops (Move/Flip would grab nothing),
            # so this role is expressed with explicit Color ops instead.
            ops.append(bgc); sels.append(sel_of(cells))      # bgc != 0 here (colors distinct)
            ops.append(0);   sels.append(sel_of(dst))
        else:
            grabbed = False
            if set(flipped) != set(cells):
                ops.append(flip_op); sels.append(sel_of(cells))   # grab group, mirror in place
                grabbed = True
            for k in range(abs(d)):
                ops.append(mv)
                if k == 0 and not grabbed:
                    sels.append(sel_of(cells))                    # first Move grabs the group
                else:
                    sels.append(sel_of([]))                       # keep same object grabbed
            if bgc != 0:
                # only the group's original footprint was zeroed by the grab
                ops.append(bgc); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
