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
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def _uni(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    lo = lb + int(round(diff_lb * (ub - lb)))
    hi = lb + int(round(diff_ub * (ub - lb)))
    lo, hi = max(lb, min(lo, ub)), max(lb, min(hi, ub))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, objc=1, otherc=2, **kw) -> dict:
    s = min(max_h, max_w, 30)   # mirrors/rotations may swap h and w
    h = _uni(diff_lb, diff_ub, 10, s)
    w = _uni(diff_lb, diff_ub, 10, s)
    objh = _uni(diff_lb, diff_ub, 2, h - 5)
    objw = _uni(diff_lb, diff_ub, 2, w - 5)
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + objh, locj:locj + objw] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [i for i, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = np.rot90(gi, 3)
    h, w = gi.shape
    rs, cs = np.where(gi == objc)
    low, left, right = rs.max(), cs.min(), cs.max()
    locis = random.randint(low + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, right))
    locje = random.randint(max(locjs + 1, left), w - 1)
    ub = min(right, locje)
    lb = max(left, locjs)
    jloc = random.randint(lb, ub)
    lnlen = locis - (low + 1)
    gib = gi.copy()
    gi = gi.copy()
    gi[locis:locie + 1, locjs:locje + 1] = otherc
    gi[low + 1:locis, jloc] = otherc
    go = gib.copy()
    go[locis - lnlen:locie + 1 - lnlen, locjs:locje + 1] = otherc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr,
           np.flipud, lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2),
           lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def _is_rect(cells):
    if not cells:
        return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit = [0, 0, hi - 1, wi - 1]
    # background = the colour dominating the grid's outer frame
    border = list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1])
    bg = Counter(int(v) for v in border).most_common(1)[0][0]
    colors = [c for c in set(I.flatten().tolist()) if c != bg]
    cellsof = {c: [(int(r), int(q)) for r, q in zip(*np.where(I == c))] for c in colors}

    # anchor = solid rectangle colour; mover = block + 1-wide stem pointing at anchor
    found = None
    for anchor in colors:
        if not _is_rect(cellsof[anchor]):
            continue
        for mover in colors:
            if mover == anchor:
                continue
            cells = cellsof[mover]
            for axis in (0, 1):  # 0: vertical stem (single-cell rows), 1: horizontal
                cnt = Counter(p[axis] for p in cells)
                stem = [p for p in cells if cnt[p[axis]] == 1]
                block = [p for p in cells if cnt[p[axis]] >= 2]
                if not stem or not _is_rect(block):
                    continue
                br0, br1, bc0, bc1 = _bbox(block)
                s0, s1 = min(p[axis] for p in stem), max(p[axis] for p in stem)
                b0, b1 = (br0, br1) if axis == 0 else (bc0, bc1)
                if s1 < b0:
                    step = -1
                elif s0 > b1:
                    step = 1
                else:
                    continue
                d = (step, 0) if axis == 0 else (0, step)
                found = dict(block=block, stem=stem, d=d, n=len(stem), color=mover)
                break
            if found:
                break
        if found:
            break

    ops, sels = [], []
    if found is None:
        return [34], [submit]
    block, stem, (dr, dc), n, colr = (found["block"], found["stem"], found["d"],
                                       found["n"], found["color"])
    dst = [(r + dr * n, c + dc * n) for r, c in block]
    sdst = set(dst)
    if colr != 0:
        # The stem is collapsed: the block slides along it toward the anchor,
        # swallowing the stem as it goes (grab once, then empty selections).
        mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(mop); sels.append(sel_of(block))
        for _ in range(n - 1):
            ops.append(mop); sels.append(sel_of([]))
        # cleanup in one stroke: stem cells the block did not land on, plus the
        # vacated original footprint (ARCLE left it at 0; repaint only if bg != 0)
        left = [p for p in stem if p not in sdst]
        if bg != 0:
            left += [p for p in block if p not in sdst]
        if left:
            ops.append(bg); sels.append(sel_of(sorted(left)))
    else:
        # colour-0 content cannot ride ARCLE's object buffer: erase stem+block, repaint
        ops.append(bg); sels.append(sel_of(sorted(set(stem) | set(block))))
        ops.append(0); sels.append(sel_of(dst))
    ops.append(34); sels.append(submit)
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
