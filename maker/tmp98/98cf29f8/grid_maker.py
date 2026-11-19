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


def sample_colors(num_examples=None) -> dict:
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, otherc) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [idx for idx, marg in enumerate(margs) if marg > 2]
    pos = choice(options)
    for k in range(pos):
        gi = rot90(gi)
    h, w = shape(gi)
    ofc = ofcolor(gi, objc)
    locis = randint(lowermost(ofc) + 2, h - 2)
    locie = randint(locis + 1, h - 1)
    locjs = randint(0, min(w - 2, rightmost(ofc)))
    locje = randint(max(locjs + 1, leftmost(ofc)), w - 1)
    otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
    ub = min(rightmost(ofc), rightmost(otherobj))
    lb = max(leftmost(ofc), leftmost(otherobj))
    jloc = randint(lb, ub)
    ln = connect((lowermost(ofc) + 1, jloc), (uppermost(otherobj) - 1, jloc))
    gib = tuple(e for e in gi)
    gi = fill(gi, otherc, otherobj)
    gi = fill(gi, otherc, ln)
    go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def _is_full_rect(cells):
    if not cells:
        return False
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _split_block(cells):
    """cells = 2x2+ rectangle plus a 1-wide stem sticking out of one side.
    Returns (rect, stem, (dr, dc)) with (dr, dc) the stem's direction, or None."""
    cells = set(cells)
    for axis in (0, 1):  # 0: stem runs vertically (split by rows), 1: horizontally
        lines = {}
        for cell in cells:
            lines.setdefault(cell[axis], []).append(cell)
        stem_keys = sorted(k for k, v in lines.items() if len(v) == 1)
        rect_keys = sorted(k for k, v in lines.items() if len(v) >= 2)
        if not stem_keys or not rect_keys:
            continue
        rect = [c for k in rect_keys for c in lines[k]]
        stem = [c for k in stem_keys for c in lines[k]]
        if not _is_full_rect(rect):
            continue
        if stem_keys == list(range(stem_keys[0], stem_keys[-1] + 1)):
            if stem_keys[0] == rect_keys[-1] + 1:
                sgn = 1
            elif stem_keys[-1] == rect_keys[0] - 1:
                sgn = -1
            else:
                continue
            step = (sgn, 0) if axis == 0 else (0, sgn)
            return rect, stem, step
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit_sel = [0, 0, hi - 1, wi - 1]
    colors = sorted(set(I.flatten().tolist()))
    cells_of = {c: [(int(r), int(q)) for r, q in zip(*np.where(I == c))] for c in colors}

    found = None
    for a in colors:                       # anchor: a solid rectangle
        if not _is_full_rect(cells_of[a]):
            continue
        for b in colors:                   # mover: rectangle + 1-wide connector stem
            if b == a:
                continue
            res = _split_block(cells_of[b])
            if res is None:
                continue
            rect, stem, (dr, dc) = res
            # stem's far end must touch the anchor
            tip = max(stem, key=lambda x: x[0] * dr + x[1] * dc)
            nxt = (tip[0] + dr, tip[1] + dc)
            if not (0 <= nxt[0] < hi and 0 <= nxt[1] < wi) or I[nxt] != a:
                continue
            others = [c for c in colors if c not in (a, b)]
            if len(others) != 1:
                continue
            found = (others[0], b, rect, stem, (dr, dc))
            break
        if found:
            break

    if found is None:
        return [34], [submit_sel]

    bg, blockc, rect, stem, (dr, dc) = found
    n = len({x[0] if dr else x[1] for x in stem})   # stem length = slide distance
    dst = {(r + dr * n, c + dc * n) for r, c in rect}
    ops, sels = [], []
    if blockc != 0:
        # Slide the block along its stem until it touches the anchor; the block
        # glides over the stem and buries the part of it inside its landing spot.
        mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(mv); sels.append(sel_of(rect))       # first Move grabs the block
        for _ in range(n - 1):
            ops.append(mv); sels.append(sel_of([]))     # keep the same grab
        # Cleanup in one stroke: the stem left sticking out beyond the block
        # (stem longer than the block is deep) plus, when bg != 0, the vacated
        # original footprint that ARCLE left at 0.
        leftover = set(stem) - dst
        if bg != 0:
            leftover |= set(rect) - dst
        if leftover:
            ops.append(int(bg)); sels.append(sel_of(sorted(leftover)))
    else:
        # colour-0 block: ARCLE's object buffer cannot carry 0 cells -> paint.
        lead = sorted(dst - set(rect))                  # band the block advances into
        trail = sorted((set(rect) | set(stem)) - dst)   # band left behind + loose stem
        ops.append(0); sels.append(sel_of(lead))
        ops.append(int(bg)); sels.append(sel_of(trail))
    ops.append(34); sels.append(submit_sel)
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
