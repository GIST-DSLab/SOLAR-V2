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
    def unif(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + int(round(d * (hi - lo)))))

    H = min(30, max_h)
    W = min(30, max_w)
    h = unif(10, max(10, H))
    w = unif(10, max(10, W))
    objh = unif(2, h - 5)
    objw = unif(2, w - 5)
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + objh, locj:locj + objw] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [i for i, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = np.rot90(gi, -1)
    h, w = gi.shape
    rs, cs = np.where(gi == objc)
    top, bot, lft, rgt = rs.min(), rs.max(), cs.min(), cs.max()
    locis = random.randint(bot + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rgt))
    locje = random.randint(max(locjs + 1, lft), w - 1)
    ub = min(rgt, locje)
    lb = max(lft, locjs)
    jloc = random.randint(lb, ub)
    ln = list(range(bot + 1, locis))
    gib = gi.copy()
    gi = gi.copy()
    gi[locis:locie + 1, locjs:locje + 1] = otherc
    for r in ln:
        gi[r, jloc] = otherc
    go = gib.copy()
    go[locis - len(ln):locie + 1 - len(ln), locjs:locje + 1] = otherc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, 3),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit_sel = [0, 0, hi - 1, wi - 1]
    ops, sels = [], []

    def cells(c):
        return [(int(r), int(cc)) for r, cc in zip(*np.where(I == c))]

    def is_rect(cs):
        rs = [r for r, _ in cs]; cc = [c for _, c in cs]
        return len(cs) == (max(rs) - min(rs) + 1) * (max(cc) - min(cc) + 1)

    present = sorted(set(I.flatten().tolist()))
    bg = anchor_c = mover_c = None
    # colour roles are fixed per episode: read them off the demonstrations.
    # anchor keeps its cell count, the mover loses its stem, background gains it.
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            if ei.shape != eo.shape:
                continue
            cols = set(ei.flatten().tolist())
            if len(cols) != 3:
                continue
            delta = {c: int((eo == c).sum()) - int((ei == c).sum()) for c in cols}
            up = [c for c in cols if delta[c] > 0]
            dn = [c for c in cols if delta[c] < 0]
            eq = [c for c in cols if delta[c] == 0]
            if len(up) == 1 and len(dn) == 1 and len(eq) == 1:
                bg, mover_c, anchor_c = up[0], dn[0], eq[0]
                break
        if bg is not None and not {bg, mover_c, anchor_c} <= set(present):
            bg = anchor_c = mover_c = None
    if bg is None:
        # fallback: structural reading of I alone
        border = np.concatenate([I[0], I[-1], I[:, 0], I[:, -1]]).tolist()
        cands = []
        for a in present:
            for m in present:
                if a == m:
                    continue
                rest = [c for c in present if c not in (a, m)]
                if len(rest) != 1:
                    continue
                if is_rect(cells(a)) and not is_rect(cells(m)):
                    cands.append((border.count(rest[0]), rest[0], a, m))
        if not cands:
            ops.append(34); sels.append(submit_sel)
            return ops, sels
        _, bg, anchor_c, mover_c = max(cands)

    A = cells(anchor_c)
    S = cells(mover_c)
    ar = [r for r, _ in A]; ac = [c for _, c in A]
    sr = [r for r, _ in S]; sc = [c for _, c in S]
    # direction the mover must travel to reach the anchor
    if max(ar) < min(sr):
        d = (-1, 0); axis = 0; order = sorted(set(sr))
    elif min(ar) > max(sr):
        d = (1, 0); axis = 0; order = sorted(set(sr), reverse=True)
    elif max(ac) < min(sc):
        d = (0, -1); axis = 1; order = sorted(set(sc))
    else:
        d = (0, 1); axis = 1; order = sorted(set(sc), reverse=True)

    # the 1-wide connector lines on the anchor side form the stem; its length is the slide
    stem = []
    for k in order:
        line = [p for p in S if p[axis] == k]
        if len(line) != 1:
            break
        stem += line
    nsteps = len(stem)
    block = sorted(set(S) - set(stem))
    dst = [(r + d[0] * nsteps, c + d[1] * nsteps) for r, c in block]

    if mover_c != 0:
        # slide the block (stem left behind as scaffolding) until it docks on the anchor
        mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[d]
        ops.append(mop); sels.append(sel_of(block))      # first Move grabs the block
        for _ in range(nsteps - 1):
            ops.append(mop); sels.append(sel_of([]))     # keep the same object grabbed
        # clean up: the block's old footprint (left at 0) and whatever part of the
        # stem the block did not slide over, i.e. old block+stem minus the block now
        leftover = sorted((set(block) | set(stem)) - set(dst))
        if bg == 0:
            leftover = [p for p in leftover if p in set(stem)]
        if leftover:
            ops.append(int(bg)); sels.append(sel_of(leftover))
    else:
        # colour-0 blocks are transparent to ARCLE's object buffer: repaint instead
        vac = sorted((set(block) | set(stem)) - set(dst))
        if vac:
            ops.append(int(bg)); sels.append(sel_of(vac))
        ops.append(0); sels.append(sel_of(dst))

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
