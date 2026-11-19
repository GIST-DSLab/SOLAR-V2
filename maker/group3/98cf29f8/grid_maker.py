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
from itertools import permutations
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def _unifint(lb_d, ub_d, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb_d * (b - a)))
    hi = a + int(round(ub_d * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, otherc=2, **kw) -> dict:
    # rotations / dmirror can swap h and w, so cap both by the smaller limit
    cap = max(10, min(30, max_h, max_w))
    h = _unifint(diff_lb, diff_ub, (10, cap))
    w = _unifint(diff_lb, diff_ub, (10, cap))
    objh = _unifint(diff_lb, diff_ub, (2, h - 5))
    objw = _unifint(diff_lb, diff_ub, (2, w - 5))
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + objh, locj:locj + objw] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [i for i, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = np.rot90(gi, -1)  # dsl rot90 (clockwise)
    h, w = gi.shape
    rr, cc = np.where(gi == objc)
    low, lft, rgt = rr.max(), cc.min(), cc.max()
    locis = random.randint(low + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rgt))
    locje = random.randint(max(locjs + 1, lft), w - 1)
    ub = min(rgt, locje)
    lb = max(lft, locjs)
    jloc = random.randint(lb, ub)
    L = locis - (low + 1)
    go = gi.copy()
    gi = gi.copy()
    gi[locis:locie + 1, locjs:locje + 1] = otherc
    gi[low + 1:locis, jloc] = otherc
    go[locis - L:locie + 1 - L, locjs:locje + 1] = otherc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def _is_rect(cells):
    if not cells:
        return False
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _analyze(I, anchor_c, mover_c):
    acells = [tuple(x) for x in np.argwhere(I == anchor_c)]
    mcells = [tuple(x) for x in np.argwhere(I == mover_c)]
    if not _is_rect(acells) or not mcells or _is_rect(mcells):
        return None
    a0, a1, b0, b1 = _bbox(acells)
    for axis in (0, 1):
        cnt = Counter(p[axis] for p in mcells)
        stem = [p for p in mcells if cnt[p[axis]] == 1]
        block = [p for p in mcells if cnt[p[axis]] != 1]
        if not stem or not _is_rect(block):
            continue
        r0, r1, c0, c1 = _bbox(block)
        if min(r1 - r0, c1 - c0) < 1:
            continue
        s = sorted(p[axis] for p in stem)
        if s != list(range(s[0], s[0] + len(s))):
            continue
        L = len(s)
        if axis == 0:
            if s[0] == r1 + 1 and a0 == s[-1] + 1:
                step = (1, 0)
            elif s[-1] == r0 - 1 and a1 == s[0] - 1:
                step = (-1, 0)
            else:
                continue
        else:
            if s[0] == c1 + 1 and b0 == s[-1] + 1:
                step = (0, 1)
            elif s[-1] == c0 - 1 and b1 == s[0] - 1:
                step = (0, -1)
            else:
                continue
        return {"block": block, "stem": stem, "step": step, "L": L}
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit_sel = [0, 0, hi - 1, wi - 1]
    colors = sorted(set(I.flatten().tolist()))
    counts = Counter(I.flatten().tolist())
    best = None
    for ac, mc in permutations(colors, 2):
        res = _analyze(I, ac, mc)
        if res is None:
            continue
        others = [c for c in colors if c not in (ac, mc)]
        if not others:
            continue
        bg = max(others, key=lambda c: counts[c])
        res.update(bg=bg, mover=mc)
        if best is None or counts[bg] > counts[best["bg"]]:
            best = res
    ops, sels = [], []
    if best is not None:
        bg, mc = best["bg"], best["mover"]
        dr, dc = best["step"]
        L = best["L"]
        # Unit of work = the whole connector object (block + stem), handled as paint:
        # 1. lift the entire stemmed object off the canvas (one connected object of one colour)
        mover_cells = sorted(best["block"] + best["stem"])
        ops.append(int(bg)); sels.append(sel_of(mover_cells))
        # 2. redraw the block where the stem pulls it: shifted by the stem length,
        #    flush against the anchor rectangle (the stem is consumed)
        dest = sorted((r + dr * L, c + dc * L) for r, c in best["block"])
        ops.append(int(mc)); sels.append(sel_of(dest))
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
