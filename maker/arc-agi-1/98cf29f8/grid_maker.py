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
def sample_colors(num_examples=None) -> dict:
    import random
    dirs = ["up", "down", "left", "right"]
    # object colors must be non-zero: ARCLE's object ops (Flip/Move) treat 0 as
    # "nothing", so a 0-coloured object could not be flipped.
    objc, otherc = random.sample(range(1, 10), 2)
    bgc = random.choice([c for c in range(10) if c not in (objc, otherc)])
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(dirs):
        examples = [{"direction": d} for d in dirs]
        examples += [{"direction": random.choice(dirs)} for _ in range(n_ex - len(dirs))]
        random.shuffle(examples)
    else:
        examples = [{"direction": d} for d in random.sample(dirs, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was demonstrated
    return {"bgc": bgc, "objc": objc, "otherc": otherc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, otherc, direction=None, **kwargs) -> dict:
    import random
    import numpy as np

    if direction is None:
        direction = random.choice(["up", "down", "left", "right"])

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = max(a, int(a + (b - a) * lb))
        hi = min(b, int(a + (b - a) * ub))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    transposed = direction in ("left", "right")
    if transposed:
        lim = max(10, min(max_h, max_w))
        hb, wb = lim, lim
    else:
        hb, wb = max(10, max_h), max(10, max_w)

    h = unifint(diff_lb, diff_ub, (10, hb))
    w = unifint(diff_lb, diff_ub, (10, wb))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))

    # anchor rectangle, kept with a bottom margin of at least 3 rows
    loci = random.randint(0, h - objh - 3)
    locj = random.randint(0, w - objw)
    rlow = loci + objh - 1
    leftm, rightm = locj, locj + objw - 1

    # moving rectangle, strictly below the anchor with a gap of >= 1 row
    locis = random.randint(rlow + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rightm))
    locje = random.randint(max(locjs + 1, leftm), w - 1)
    jloc = random.randint(max(leftm, locjs), min(rightm, locje))
    L = locis - rlow - 1  # length of the connecting line

    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + objh, locj:locj + objw] = objc
    go = gi.copy()
    gi[locis:locie + 1, locjs:locje + 1] = otherc
    gi[rlow + 1:locis, jloc] = otherc
    go[locis - L:locie + 1 - L, locjs:locje + 1] = otherc

    if direction == "down":
        gi, go = np.flipud(gi), np.flipud(go)
    elif direction == "left":
        gi, go = gi.T, go.T
    elif direction == "right":
        gi, go = np.fliplr(gi.T), np.fliplr(go.T)

    # perpendicular flip: extra variety without changing the movement direction
    if random.random() < 0.5:
        if direction in ("up", "down"):
            gi, go = np.fliplr(gi), np.fliplr(go)
        else:
            gi, go = np.flipud(gi), np.flipud(go)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    import numpy as np
    from itertools import permutations
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    colors = sorted(set(int(v) for v in I.flatten().tolist()))

    def cellset(col):
        rs, cs = np.nonzero(I == col)
        return set(zip(rs.tolist(), cs.tolist()))

    def bbox(cs):
        rr = [r for r, _ in cs]
        cc = [c for _, c in cs]
        return min(rr), max(rr), min(cc), max(cc)

    def solid(cs):
        if len(cs) < 4:
            return False
        r0, r1, c0, c1 = bbox(cs)
        return (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2 and len(cs) == (r1 - r0 + 1) * (c1 - c0 + 1)

    best = None
    for bg, anc, mov in permutations(colors, 3):
        anchor = cellset(anc)
        movcells = cellset(mov)
        if not solid(anchor) or not movcells:
            continue

        def isbg(r, c, _bg=bg):
            return (not (0 <= r < h and 0 <= c < w)) or I[r, c] == _bg

        # a line cell is 1 wide: background on both sides across the line
        line = {(r, c) for (r, c) in movcells
                if (isbg(r, c - 1) and isbg(r, c + 1)) or (isbg(r - 1, c) and isbg(r + 1, c))}
        rect = movcells - line
        if not line or not solid(rect):
            continue

        r0, r1, c0, c1 = bbox(rect)
        ar0, ar1, ac0, ac1 = bbox(anchor)
        L = len(line)
        lrows = sorted({r for r, _ in line})
        lcols = sorted({c for _, c in line})

        direction = None
        if len(lcols) == 1 and c0 <= lcols[0] <= c1 and lrows == list(range(lrows[0], lrows[0] + L)):
            if lrows[-1] == r0 - 1 and ar1 == lrows[0] - 1:
                direction = "up"
            elif lrows[0] == r1 + 1 and ar0 == lrows[-1] + 1:
                direction = "down"
        elif len(lrows) == 1 and r0 <= lrows[0] <= r1 and lcols == list(range(lcols[0], lcols[0] + L)):
            if lcols[-1] == c0 - 1 and ac1 == lcols[0] - 1:
                direction = "left"
            elif lcols[0] == c1 + 1 and ac0 == lcols[-1] + 1:
                direction = "right"
        if direction is None:
            continue

        # the strip = connecting line + moving rectangle; reflecting it across the
        # strip's middle carries the rectangle up against the anchor.
        if direction == "up":
            rr0, rr1, cc0, cc1, flip = r0 - L, r1, c0, c1, 27
        elif direction == "down":
            rr0, rr1, cc0, cc1, flip = r0, r1 + L, c0, c1, 27
        elif direction == "left":
            rr0, rr1, cc0, cc1, flip = r0, r1, c0 - L, c1, 26
        else:
            rr0, rr1, cc0, cc1, flip = r0, r1, c0, c1 + L, 26

        G = I.copy()
        sub = G[rr0:rr1 + 1, cc0:cc1 + 1]
        G[rr0:rr1 + 1, cc0:cc1 + 1] = np.flipud(sub) if flip == 27 else np.fliplr(sub)
        if flip == 27:
            moved_line = sorted({(rr0 + rr1 - r, c) for r, c in line})
        else:
            moved_line = sorted({(r, cc0 + cc1 - c) for r, c in line})
        for r, c in moved_line:
            G[r, c] = bg
        if G.shape == O.shape and np.array_equal(G, O):
            best = (flip, rr0, rr1, cc0, cc1, moved_line, bg)
            break

    if best is not None:
        flip, rr0, rr1, cc0, cc1, moved_line, bg = best
        # selection is exactly the full rectangular strip (background included) —
        # the reflection acts on the whole strip, not on a non-rectangular object.
        strip = [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]
        ops.append(flip)
        sels.append(sel_of(strip))
        # what the reflection leaves behind: the line, now on the far side — erase it
        ops.append(int(bg))
        sels.append(sel_of(moved_line))
    else:
        by_color = {}
        for r in range(min(h, ho)):
            for c in range(min(w, wo)):
                if I[r, c] != O[r, c]:
                    by_color.setdefault(int(O[r, c]), []).append((r, c))
        for col, cs in by_color.items():
            ops.append(col)
            sels.append(sel_of(cs))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # full grid rectangle
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
