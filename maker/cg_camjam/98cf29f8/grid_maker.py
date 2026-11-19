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


def _canvas(c, h, w):
    return [[c] * w for _ in range(h)]


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, otherc) -> dict:
    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))
    H = min(30, max_h); W = min(30, max_w)
    h = unif(10, H); w = unif(10, W)
    objh = unif(2, h - 5); objw = unif(2, w - 5)
    loci = random.randint(0, h - objh); locj = random.randint(0, w - objw)
    gi = _canvas(bgc, h, w)
    for r in range(loci, loci + objh):
        for c in range(locj, locj + objw):
            gi[r][c] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    pos = random.choice([i for i, m in enumerate(margs) if m > 2])
    for _ in range(pos):
        gi = _rot90(gi)
    h, w = len(gi), len(gi[0])
    ofc = [(r, c) for r in range(h) for c in range(w) if gi[r][c] == objc]
    lo = max(r for r, _ in ofc); ri = max(c for _, c in ofc); le = min(c for _, c in ofc)
    locis = random.randint(lo + 2, h - 2); locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, ri)); locje = random.randint(max(locjs + 1, le), w - 1)
    jloc = random.randint(max(le, locjs), min(ri, locje))
    ln = [(r, jloc) for r in range(lo + 1, locis)]
    go = [row[:] for row in gi]
    for r in range(locis, locie + 1):
        for c in range(locjs, locje + 1):
            gi[r][c] = otherc
            go[r - len(ln)][c] = otherc
    for r, c in ln:
        gi[r][c] = otherc
    gi = np.array(gi); go = np.array(go)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find(I):
    colors = sorted(set(I.flatten().tolist()))
    if len(colors) != 3:
        return None
    cells = {c: {(int(r), int(q)) for r, q in zip(*np.where(I == c))} for c in colors}

    def is_rect(s):
        rs = [r for r, _ in s]; cs = [c for _, c in s]
        return len(s) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    for a in colors:                      # anchor: a solid rectangle
        if not is_rect(cells[a]):
            continue
        for b in colors:                  # block colour: rectangle + 1-wide stem
            if b == a:
                continue
            B = cells[b]
            touch = set()
            for (r, c) in cells[a]:
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (r + d[0], c + d[1]) in B:
                        touch.add(((r + d[0], c + d[1]), d))
            if len(touch) != 1:
                continue
            (s, d), = touch
            perp = [(d[1], d[0]), (-d[1], -d[0])]
            stem = []
            cur = s
            while cur in B and not any((cur[0] + p[0], cur[1] + p[1]) in B for p in perp):
                stem.append(cur)
                cur = (cur[0] + d[0], cur[1] + d[1])
            if not stem or cur not in B:
                continue
            rect = B - set(stem)
            if not is_rect(rect):
                continue
            bg = [c for c in colors if c not in (a, b)][0]
            return dict(bg=bg, block=b, rect=sorted(rect), stem=stem,
                        step=(-d[0], -d[1]), nsteps=len(stem))
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]
    f = _find(I)
    if f is None:
        return [34], [submit_sel]
    bg, blockc, rect = f["bg"], f["block"], f["rect"]
    stem = set(f["stem"])
    dr, dc = f["step"]; n = f["nsteps"]
    dst = [(r + dr * n, c + dc * n) for r, c in rect]

    # length of the block measured along the slide direction
    along = len({(r if dr else c) for r, c in rect})
    # The stem collapses as the block closes the gap. If the stem is longer than
    # the block, the docked block cannot cover all of it: erase the scaffolding.
    if n > along:
        ops.append(int(bg)); sels.append(sel_of(sorted(stem)))

    if blockc != 0:
        # the block slides back along its stem until it docks on the anchor
        mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(mop); sels.append(sel_of(rect))      # first Move grabs the block
        for _ in range(n - 1):
            ops.append(mop); sels.append(sel_of([]))    # keep the same block grabbed
        hole = sorted(set(rect) - set(dst))
        if bg != 0 and hole:
            ops.append(int(bg)); sels.append(sel_of(hole))
    else:
        # colour-0 block can't be carried by ARCLE's object buffer: clear the part
        # of the block beyond its docked position, then fill the docked footprint
        # (only cells not already colour 0 at this moment).
        g = I.copy()
        for r, c in (stem if n > along else []):
            g[r, c] = bg
        vac = sorted(set(rect) - set(dst))
        ops.append(int(bg)); sels.append(sel_of(vac))
        for r, c in vac:
            g[r, c] = bg
        gap = sorted(p for p in dst if g[p] != 0)
        if gap:
            ops.append(0); sels.append(sel_of(gap))

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
