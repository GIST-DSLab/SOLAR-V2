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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def _rot90(g): return [list(r) for r in zip(*g[::-1])]
def _rot180(g): return [list(r)[::-1] for r in g[::-1]]
def _rot270(g): return [list(r) for r in zip(*g)][::-1]
def _dmirror(g): return [list(r) for r in zip(*g)]
def _cmirror(g): return _rot180(_dmirror(g))
def _vmirror(g): return [list(r)[::-1] for r in g]
def _hmirror(g): return [list(r) for r in g[::-1]]
def _ident(g): return [list(r) for r in g]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, objc=1, otherc=2) -> dict:
    def unifint(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))
    h = unifint(10, max(10, min(30, max_h)))
    w = unifint(10, max(10, min(30, max_w)))
    objh = unifint(2, h - 5)
    objw = unifint(2, w - 5)
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + objh):
        for c in range(locj, locj + objw):
            gi[r][c] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [i for i, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = _rot90(gi)
    h, w = len(gi), len(gi[0])
    ofc = [(r, c) for r in range(h) for c in range(w) if gi[r][c] == objc]
    lo = max(r for r, _ in ofc)
    ri = max(c for _, c in ofc)
    le = min(c for _, c in ofc)
    locis = random.randint(lo + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, ri))
    locje = random.randint(max(locjs + 1, le), w - 1)
    ub = min(ri, locje)
    lb = max(le, locjs)
    jloc = random.randint(lb, ub)
    ln = [(r, jloc) for r in range(lo + 1, locis)]
    go = [list(r) for r in gi]
    for r in range(locis, locie + 1):
        for c in range(locjs, locje + 1):
            gi[r][c] = otherc
            go[r - len(ln)][c] = otherc
    for r, c in ln:
        gi[r][c] = otherc
    mfs = [_ident, _dmirror, _cmirror, _vmirror, _hmirror, _rot90, _rot180, _rot270]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def _find(I):
    hi, wi = len(I), len(I[0])
    border = Counter(I[r][c] for r in range(hi) for c in range(wi)
                     if r in (0, hi - 1) or c in (0, wi - 1))
    total = Counter(v for row in I for v in row)
    # background = colour framing the canvas; if that is ambiguous, the one under which
    # the remaining two colours parse as (anchor rectangle, block + 1-wide stem)
    for bg in sorted(total, key=lambda k: (-border[k], -total[k])):
        res = _find_bg(I, bg)
        if res is not None:
            return res
    return None


def _find_bg(I, bg):
    hi, wi = len(I), len(I[0])
    cells = {}
    for r in range(hi):
        for c in range(wi):
            if I[r][c] != bg:
                cells.setdefault(I[r][c], set()).add((r, c))
    if len(cells) != 2:
        return None

    def is_rect(S):
        rs = [r for r, _ in S]
        cs = [c for _, c in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    anchors = [k for k, S in cells.items() if is_rect(S)]
    if len(anchors) != 1:
        return None
    ac = anchors[0]
    mc = [k for k in cells if k != ac][0]
    S = cells[mc]
    for axis in (0, 1):  # 0: stem vertical (split by rows), 1: stem horizontal (split by cols)
        cnt = Counter(p[axis] for p in S)
        block = {p for p in S if cnt[p[axis]] >= 2}
        stem = S - block
        if not block or not stem or not is_rect(block):
            continue
        if len({p[1 - axis] for p in stem}) != 1:
            continue
        bmin = min(p[axis] for p in block)
        smin = min(p[axis] for p in stem)
        sgn = 1 if smin > bmin else -1
        step = (sgn, 0) if axis == 0 else (0, sgn)
        return dict(bg=bg, mc=mc, block=sorted(block), stem=sorted(stem),
                    step=step, n=len(stem))
    return None


def derive_operations(I, O=None, examples=None):
    I = [list(map(int, r)) for r in I]
    hi, wi = len(I), len(I[0])
    submit = [0, 0, hi - 1, wi - 1]  # whole grid for Submit
    f = _find(I)
    if f is None:
        return [34], [submit]
    bg, mc, block, stem = f["bg"], f["mc"], f["block"], f["stem"]
    (dr, dc), n = f["step"], f["n"]
    ops, sels = [], []

    # 1. the stem is a guide line (scaffolding): clear it back to background
    ops.append(bg)
    sels.append(sel_of(stem))

    dst = [(r + dr * n, c + dc * n) for r, c in block]
    if mc != 0:
        # 2. slide the block along the stem's track until it touches the anchor
        mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(mv)
        sels.append(sel_of(block))      # first Move grabs the block
        for _ in range(n - 1):
            ops.append(mv)
            sels.append(sel_of([]))     # keep the same object grabbed
        # 3. restore the block's vacated original footprint
        hole = sorted(set(block) - set(dst))
        if bg != 0 and hole:
            ops.append(bg)
            sels.append(sel_of(hole))
    else:
        # colour-0 blocks cannot be carried by ARCLE's object buffer: paint instead
        vac = sorted(set(block) - set(dst))
        if vac:
            ops.append(bg)
            sels.append(sel_of(vac))
        ops.append(0)
        sels.append(sel_of(dst))

    ops.append(34)
    sels.append(submit)
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
