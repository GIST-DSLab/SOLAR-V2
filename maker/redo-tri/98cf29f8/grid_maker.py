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
from maker.sel_helpers import sel_of

# ---------------------------------------------------------------------------
# Task 98cf29f8
#   I : a solid rectangle ANCHOR (colour objc) + a solid rectangle BLOCK
#       (colour otherc) that is tethered to the anchor by a 1-wide LINE of the
#       same colour otherc.
#   O : the BLOCK has slid along that line (by exactly the line's length) until
#       it abuts the anchor; the line itself is gone.
#   => pure translation of the block  +  cleanup of what it left behind.
# ---------------------------------------------------------------------------

ALL_TRANSFORMS = ["identity", "fliplr", "rot180", "flipud",
                  "rot90cw", "antitranspose", "rot90ccw", "transpose"]
# one representative per movement direction (up / down / right / left)
CORE_TRANSFORMS = ["identity", "rot180", "rot90cw", "rot90ccw"]
SWAP_TRANSFORMS = {"rot90cw", "rot90ccw", "transpose", "antitranspose"}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, otherc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(CORE_TRANSFORMS):
        examples = [{"transform": t} for t in CORE_TRANSFORMS]
        examples += [{"transform": random.choice(ALL_TRANSFORMS)}
                     for _ in range(n_ex - len(CORE_TRANSFORMS))]
        random.shuffle(examples)
    else:
        examples = [{"transform": t}
                    for t in random.sample(CORE_TRANSFORMS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "otherc": otherc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, otherc,
             transform=None) -> dict:
    def unifint(dlb, dub, bounds):
        a, b = bounds
        if b < a:
            b = a
        ir = b - a
        lo = a + int(dlb * ir)
        hi = a + int(dub * ir)
        lo = max(a, min(b, lo))
        hi = max(lo, min(b, hi))
        return random.randint(lo, hi)

    def apply_tf(g, name):
        a = np.array(g, dtype=int)
        if name == "fliplr":
            a = np.fliplr(a)
        elif name == "flipud":
            a = np.flipud(a)
        elif name == "rot180":
            a = np.rot90(a, 2)
        elif name == "rot90cw":
            a = np.rot90(a, 3)
        elif name == "rot90ccw":
            a = np.rot90(a, 1)
        elif name == "transpose":
            a = a.T
        elif name == "antitranspose":
            a = np.rot90(a, 2).T
        return a

    if transform is None:
        transform = random.choice(ALL_TRANSFORMS)
    swap = transform in SWAP_TRANSFORMS

    hub = max(10, min(30, max_w if swap else max_h))
    wub = max(10, min(30, max_h if swap else max_w))

    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))

    # anchor placed so that at least 3 rows of room remain below it
    loci = random.randint(0, h - objh - 3)
    locj = random.randint(0, w - objw)
    lowermost = loci + objh - 1
    leftmost = locj
    rightmost = locj + objw - 1

    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + objh):
        for c in range(locj, locj + objw):
            gi[r][c] = objc
    go = [row[:] for row in gi]

    locis = random.randint(lowermost + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rightmost))
    locje = random.randint(max(locjs + 1, leftmost), w - 1)
    jloc = random.randint(max(leftmost, locjs), min(rightmost, locje))
    lnlen = locis - lowermost - 1          # length of the tether line (>= 1)

    # input: block + tether line
    for r in range(locis, locie + 1):
        for c in range(locjs, locje + 1):
            gi[r][c] = otherc
    for r in range(lowermost + 1, locis):
        gi[r][jloc] = otherc

    # output: block slid up by the line length, line gone
    for r in range(locis - lnlen, locie + 1 - lnlen):
        for c in range(locjs, locje + 1):
            go[r][c] = otherc

    gi_t = apply_tf(gi, transform)
    go_t = apply_tf(go, transform)

    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi_t),
        "output": tuple(tuple(int(v) for v in row) for row in go_t),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    colors = sorted(set(I.flatten().tolist()))
    cells_of = {c: set() for c in colors}
    for r in range(hi):
        for c in range(wi):
            cells_of[int(I[r, c])].add((r, c))

    def is_full_rect(S):
        rs = [p[0] for p in S]
        cs = [p[1] for p in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    def split_rect_line(S):
        """S == solid rectangle (>=2 wide) + straight 1-wide line stuck to one
        of its sides.  Returns (rect_cells, line_cells, dir_rect_to_line)."""
        for transposed in (False, True):
            T = {(c, r) for (r, c) in S} if transposed else set(S)
            rows = {}
            for r, c in T:
                rows.setdefault(r, []).append(c)
            line_rows = sorted(r for r in rows if len(rows[r]) == 1)
            rect_rows = sorted(r for r in rows if len(rows[r]) > 1)
            if not line_rows or not rect_rows:
                continue
            if rect_rows != list(range(rect_rows[0], rect_rows[-1] + 1)):
                continue
            span, ok = None, True
            for r in rect_rows:
                cs = sorted(rows[r])
                if cs != list(range(cs[0], cs[-1] + 1)):
                    ok = False
                    break
                if span is None:
                    span = (cs[0], cs[-1])
                elif span != (cs[0], cs[-1]):
                    ok = False
                    break
            if not ok:
                continue
            if line_rows != list(range(line_rows[0], line_rows[-1] + 1)):
                continue
            jset = {rows[r][0] for r in line_rows}
            if len(jset) != 1:
                continue
            j = jset.pop()
            if not (span[0] <= j <= span[1]):
                continue
            if line_rows[-1] + 1 == rect_rows[0]:
                step = -1
            elif rect_rows[-1] + 1 == line_rows[0]:
                step = 1
            else:
                continue
            rect = {(r, c) for r in rect_rows for c in range(span[0], span[1] + 1)}
            line = {(r, j) for r in line_rows}
            if transposed:
                rect = {(c, r) for (r, c) in rect}
                line = {(c, r) for (r, c) in line}
                d = (0, step)
            else:
                d = (step, 0)
            return rect, line, d
        return None

    # --- identify block (rect + tether), anchor rectangle, background --------
    found = None
    for M in sorted(colors, key=lambda c: len(cells_of[c])):
        sp = split_rect_line(cells_of[M])
        if sp is None:
            continue
        rect, line, d = sp
        far = max(line, key=lambda p: p[0] * d[0] + p[1] * d[1])
        nxt = (far[0] + d[0], far[1] + d[1])
        for A in colors:
            if A == M or not cells_of[A]:
                continue
            if A != M and is_full_rect(cells_of[A]) and nxt in cells_of[A]:
                found = (M, A, rect, line, d)
                break
        if found:
            break

    ops, sels = [], []

    if found is None:                       # defensive fallback (should not fire)
        for r in range(min(hi, ho)):
            for c in range(min(wi, wo)):
                if I[r, c] != O[r, c]:
                    ops.append(int(O[r, c]))
                    sels.append(sel_of([(r, c)]))
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    M, A, rect, line, d = found
    rest = [c for c in colors if c not in (M, A)]
    bgc = max(rest, key=lambda c: len(cells_of[c])) if rest else 0

    steps = len(line)                       # slide exactly the tether's length
    move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[d]

    src = sorted(rect)
    cur = list(src)

    # grab the block once, then keep sliding with an empty selection
    ops.append(move_op)
    sels.append(sel_of(src))
    cur = [(r + d[0], c + d[1]) for r, c in cur]
    for _ in range(steps - 1):
        ops.append(move_op)
        sels.append(sel_of([]))
        cur = [(r + d[0], c + d[1]) for r, c in cur]

    dest = set(cur)
    src_set = set(src)
    line_set = set(line)

    # cleanup: the block's vacated footprint (left at 0 by the grab) plus the
    # part of the tether line the block did not end up covering.
    cleanup = []
    for cell in sorted((src_set | line_set) - dest):
        now = 0 if cell in src_set else M
        if now != bgc:
            cleanup.append(cell)
    if cleanup:
        ops.append(int(bgc))
        sels.append(sel_of(cleanup))

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
