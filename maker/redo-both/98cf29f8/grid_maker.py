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

DIRECTIONS = ["up", "down", "left", "right"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c != bgc])
    # the sliding object must be non-zero: ARCLE's Move only grabs non-zero cells
    otherc = random.choice([c for c in cols if c not in (bgc, objc) and c != 0])
    n_ex = num_examples if num_examples else 3
    variants = [{"direction": d} for d in DIRECTIONS]
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
        direction = random.choice(DIRECTIONS)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    # how each mirror/rotation maps a motion vector (dr, dc)
    vfs = (lambda v: (v[0], v[1]), lambda v: (v[1], v[0]), lambda v: (-v[1], -v[0]),
           lambda v: (v[0], -v[1]), lambda v: (-v[0], v[1]), lambda v: (v[1], -v[0]),
           lambda v: (-v[0], -v[1]), lambda v: (-v[1], v[0]))
    names = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    canon = {"up": (0,), "down": (4,), "left": (1,), "right": (2,)}

    def build():
        lim = max(10, min(max_h, max_w))  # a transposing mirror may swap h and w
        h = unifint(diff_lb, diff_ub, (10, lim))
        w = unifint(diff_lb, diff_ub, (10, lim))
        objh = unifint(diff_lb, diff_ub, (2, h - 5))
        objw = unifint(diff_lb, diff_ub, (2, w - 5))
        loci = random.randint(0, h - objh)
        locj = random.randint(0, w - objw)
        obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
        gi = canvas(bgc, (h, w))
        gi = fill(gi, objc, obj)
        margs = (h - (loci + objh), w - (locj + objw), loci, locj)
        pos = random.choice([idx for idx, m in enumerate(margs) if m > 2])
        for _ in range(pos):
            gi = rot90(gi)
        hh, ww = shape(gi)
        ofc = ofcolor(gi, objc)
        locis = random.randint(lowermost(ofc) + 2, hh - 2)
        locie = random.randint(locis + 1, hh - 1)
        locjs = random.randint(0, min(ww - 2, rightmost(ofc)))
        locje = random.randint(max(locjs + 1, leftmost(ofc)), ww - 1)
        otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
        jloc = random.randint(max(leftmost(ofc), leftmost(otherobj)),
                              min(rightmost(ofc), rightmost(otherobj)))
        ln = connect((lowermost(ofc) + 1, jloc), (uppermost(otherobj) - 1, jloc))
        gib = tuple(e for e in gi)
        gi = fill(gi, otherc, otherobj)
        gi = fill(gi, otherc, ln)
        go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
        chosen = None
        for _ in range(64):
            idxs = random.sample(range(8), random.choice((1, 2)))
            v = (-1, 0)
            for i in idxs:
                v = vfs[i](v)
            if names[v] == direction:
                chosen = idxs
                break
        if chosen is None:
            chosen = canon[direction]
        for i in chosen:
            gi = mfs[i](gi)
            go = mfs[i](go)
        return {"input": gi, "output": go}

    res = build()
    for _ in range(200):
        g = res["input"]
        hh, ww = len(g), len(g[0])
        ring = ([g[0][c] for c in range(ww)] + [g[hh - 1][c] for c in range(ww)] +
                [g[r][0] for r in range(1, hh - 1)] + [g[r][ww - 1] for r in range(1, hh - 1)])
        if ring.count(bgc) > max(ring.count(objc), ring.count(otherc)):
            break
        res = build()
    return res


def derive_operations(I, O):
    """Rule (measured from I alone): a solid block of one colour, and a second colour
    forming a solid rectangle plus a 1-cell-wide stalk pointing at the block.  Slide the
    rectangle along the stalk until it touches the block; the stalk disappears."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def cells_of(col):
        rr, cc = np.where(I == col)
        return [(int(a), int(b)) for a, b in zip(rr, cc)]

    def bbox(cs):
        rs = [p[0] for p in cs]
        ks = [p[1] for p in cs]
        return min(rs), max(rs), min(ks), max(ks)

    colors = sorted(set(int(x) for x in I.flatten().tolist()))
    ring = ([int(I[0, c]) for c in range(w)] + [int(I[h - 1, c]) for c in range(w)] +
            [int(I[r, 0]) for r in range(1, h - 1)] + [int(I[r, w - 1]) for r in range(1, h - 1)])
    ring_bg = Counter(ring).most_common(1)[0][0]

    def analyze(strict):
        found = []
        for bg in colors:
            rest = [c for c in colors if c != bg]
            if len(rest) != 2:
                continue
            for blockc, otherc in ((rest[0], rest[1]), (rest[1], rest[0])):
                bcells = cells_of(blockc)
                ocells = cells_of(otherc)
                if not bcells or not ocells:
                    continue
                br0, br1, bc0, bc1 = bbox(bcells)
                if len(bcells) != (br1 - br0 + 1) * (bc1 - bc0 + 1):
                    continue  # the anchor block must be a filled rectangle

                def val(r, c, _bg=bg):
                    return int(I[r, c]) if 0 <= r < h and 0 <= c < w else _bg

                # stalk cells: object colour with background on both sides of one axis
                line = set()
                for (r, c) in ocells:
                    if (val(r, c - 1) == bg and val(r, c + 1) == bg) or \
                       (val(r - 1, c) == bg and val(r + 1, c) == bg):
                        line.add((r, c))
                rect = sorted(set(ocells) - line)
                if not line or not rect:
                    continue
                r0, r1, c0, c1 = bbox(rect)
                if len(rect) != (r1 - r0 + 1) * (c1 - c0 + 1):
                    continue  # the movable body must be a filled rectangle
                if strict and (r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2):
                    continue
                if len(set(p[0] for p in line)) != 1 and len(set(p[1] for p in line)) != 1:
                    continue  # the stalk must be straight
                cols_ov = not (c1 < bc0 or bc1 < c0)
                rows_ov = not (r1 < br0 or br1 < r0)
                if cols_ov and not rows_ov:
                    if br1 < r0:
                        mop, steps, d = 20, r0 - br1 - 1, (-1, 0)
                    elif br0 > r1:
                        mop, steps, d = 21, br0 - r1 - 1, (1, 0)
                    else:
                        continue
                elif rows_ov and not cols_ov:
                    if bc1 < c0:
                        mop, steps, d = 23, c0 - bc1 - 1, (0, -1)
                    elif bc0 > c1:
                        mop, steps, d = 22, bc0 - c1 - 1, (0, 1)
                    else:
                        continue
                else:
                    continue
                if steps < 1:
                    continue
                if strict and steps != len(line):
                    continue  # the stalk spans exactly the gap it points across
                found.append((1 if bg == ring_bg else 0, bg, rect, sorted(line), mop, steps, d))
        if not found:
            return None
        found.sort(key=lambda t: -t[0])
        return found[0]

    plan = analyze(True) or analyze(False)

    if plan is not None:
        _, bg, rect, line, mop, steps, d = plan
        # slide the body one cell at a time until it touches the block
        cur = list(rect)
        ops.append(mop)
        sels.append(sel_of(cur))                     # first Move grabs the body
        cur = [(r + d[0], c + d[1]) for r, c in cur]
        for _ in range(steps - 1):
            ops.append(mop)
            sels.append(sel_of([]))                  # empty -> keep the same object grabbed
            cur = [(r + d[0], c + d[1]) for r, c in cur]
        dst = set(cur)
        # clean-up: the stalk part the body no longer covers, plus its vacated footprint
        repair = set(line) - dst
        if bg != 0:
            repair |= (set(rect) - dst)
        if repair:
            ops.append(int(bg))
            sels.append(sel_of(sorted(repair)))

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
