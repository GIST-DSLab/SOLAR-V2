"""
ARC Task: b2862040 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


# ── helpers (RE-ARC DSL style, self-contained) ───────────────────────────────

def _box(oh, ow):
    cells = set()
    for j in range(ow):
        cells.add((0, j)); cells.add((oh - 1, j))
    for i in range(oh):
        cells.add((i, 0)); cells.add((i, ow - 1))
    return cells


def _corners(obj):
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _dneighbors(ij):
    r, c = ij
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _neighbors(ij):
    r, c = ij
    return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _outbox(obj):
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
    out = set()
    for j in range(c0, c1 + 1):
        out.add((r0, j)); out.add((r1, j))
    for i in range(r0, r1 + 1):
        out.add((i, c0)); out.add((i, c1))
    return out


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    lo_i = int(round(lo)); hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def _components(G):
    h, w = G.shape
    comp_id = -np.ones((h, w), dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if comp_id[r, c] != -1:
                continue
            col = int(G[r, c])
            cid = len(comps)
            cells = []
            q = deque([(r, c)])
            comp_id[r, c] = cid
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and comp_id[ny, nx] == -1 and G[ny, nx] == col:
                        comp_id[ny, nx] = cid
                        q.append((ny, nx))
            comps.append((col, cells))
    return comp_id, comps


def _apply_rule(G, bgc):
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    comp_id, comps = _components(G)
    pocket = np.zeros((h, w), dtype=bool)
    for col, cells in comps:
        if col != bgc:
            continue
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        for r, c in cells:
            pocket[r, c] = True
    targets = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if not pocket[r, c]:
                continue
            for ny, nx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= ny < h and 0 <= nx < w and G[ny, nx] != bgc:
                    tid = comp_id[ny, nx]
                    if tid not in seen:
                        seen.add(tid)
                        targets.append(tid)
    return comps, targets, pocket.any()


# ── 1. sample_colors ─────────────────────────────────────────────────────────

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ── 2. generate ──────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    while True:
        h = _unifint(diff_lb, diff_ub, (10, max_h))
        w = _unifint(diff_lb, diff_ub, (10, max_w))
        nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        gi = np.full((h, w), bgc, dtype=int)
        inds = {(i, j) for i in range(h) for j in range(w)}
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6)
            ow = random.randint(3, 6)
            obj = _box(oh, ow)
            if random.choice((True, False)):
                nkeep = _unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                obj = set(obj)
                obj.discard(random.choice(sorted(obj - _corners(obj))))
                for _ in range(nrem - 1):
                    xx = [ij for ij in obj if len(_dneighbors(ij) & obj) == 1]
                    if len(xx) == 0:
                        break
                    obj.discard(random.choice(sorted(xx)))
            npert = _unifint(diff_lb, diff_ub, (0, oh + ow))
            o1 = _outbox(obj)
            o2 = _outbox(o1)
            o3 = _outbox(o2)
            objcands = o1 | o2 | o3
            obj = set(obj)
            for _ in range(npert):
                nb = set()
                for ij in obj:
                    nb |= _dneighbors(ij)
                opts = sorted((objcands - obj) & nb)
                if not opts:
                    break
                obj.add(random.choice(opts))
            mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
            obj = {(r - mr, c - mc) for r, c in obj}
            oh = max(r for r, _ in obj) + 1
            ow = max(c for _, c in obj) + 1
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if len(cands) == 0:
                continue
            loc = random.choice(sorted(cands))
            plcd = {(r + loc[0], c + loc[1]) for r, c in obj}
            if plcd.issubset(inds):
                col = random.choice(remcols)
                for r, c in plcd:
                    gi[r, c] = col
                succ += 1
                nb = set()
                for ij in plcd:
                    nb |= _neighbors(ij)
                inds = (inds - plcd) - nb
        comps, targets, has_pocket = _apply_rule(gi, bgc)
        if has_pocket:
            go = gi.copy()
            for tid in targets:
                for r, c in comps[tid][1]:
                    go[r, c] = 8
            return {"input": gi.tolist(), "output": go.tolist()}


# ── 3. derive_operations ─────────────────────────────────────────────────────

def derive_operations(I, O, examples=None):
    """
    Rule (from I and the demonstrations only):
      - bgc = the canvas colour the generator paints first; confirmed as the
        dominant colour pooled over I and every example input (all share it).
      - enclosed pockets = 4-connected bgc components not touching the border.
      - every non-bgc 4-connected same-colour object orthogonally adjacent to a
        pocket turns 8.
    All such objects are recoloured together in ONE Color8 stroke whose
    selection is "the objects that touch an enclosed pocket" -- named from I
    alone. O is used only to size the Submit selection.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = ex[0] if not isinstance(ex, dict) else ex.get("input")
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    comps, targets, _ = _apply_rule(I, bgc)

    ops, sels = [], []
    cells = []
    for tid in targets:
        col, ccells = comps[tid]
        if col != 8:
            cells.extend(ccells)
    if cells:
        ops.append(8)                       # Color8 on all pocket-bounding objects
        sels.append(sel_of(sorted(set(cells))))

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
                        f"num_examples+1 ({num_examples + 1}) for task b2862040"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b2862040"
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
                                f"for task b2862040"
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
                    f"Failed to build a complete episode for task b2862040 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b2862040-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
