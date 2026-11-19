"""
ARC Task: b8825c91 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(15, min(max_h, max_w) // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx = bx - {ch}
    for (i, j), v in obj.items():
        g[i, j] = v
    # paint tr: cells of dmirror(g) with col>=row
    g2 = g.copy()
    dm = g.T
    for i in range(h):
        for j in range(w):
            if j >= i:
                g2[i, j] = dm[i, j]
    g = g2
    g = np.hstack([g, g[:, ::-1]])
    g = np.vstack([g, g[::-1, :]])
    go = g.copy()
    gi = g.copy()
    for alph in (2, 1):
        locidev = unifint(diff_lb, diff_ub, (1, alph * h))
        locjdev = unifint(diff_lb, diff_ub, (1, w))
        loci = alph * h - locidev
        locj = w - locjdev
        loci2 = unifint(diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
        gi[loci:loci2 + 1, locj:locj2 + 1] = 4
        gi = gi[::-1, ::-1].copy()
        go = go[::-1, ::-1].copy()
    mfs = [lambda x: x, lambda x: x.T, lambda x: x[::-1, ::-1].T,
           lambda x: x[:, ::-1], lambda x: x[::-1, :],
           lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the picture is invariant under every mirror
    # of the square, so each cell equals any visible member of its mirror orbit; colour 4
    # only marks "unknown".  The WHOLE GRID is the unit: mark the 4-cells empty (0, which
    # Paste treats as transparent), then for each mirror T overlay the grid onto its own
    # mirror image: CopyO whole grid, apply T to the whole grid, Paste the copy back at
    # (0,0) so known cells sit on top and the empties show T's content.
    # A real colour-0 cell has an all-0 orbit, so the 0 placeholder never corrupts it.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    whole = [0, 0, H - 1, W - 1]          # full-grid rectangle (whole picture is the unit)
    ops, sels = [], []
    G = I.copy()

    hidden = [(r, c) for r in range(H) for c in range(W) if I[r, c] == 4]
    if not hidden:
        ops.append(34); sels.append(whole)
        return ops, sels
    ops.append(0); sels.append(sel_of(hidden))   # all colour-4 cells -> empty
    G[I == 4] = 0

    # the mirrors of the picture, each as whole-grid ARCLE ops
    FN = {24: lambda x: np.rot90(x, 1), 25: lambda x: np.rot90(x, 3),
          26: np.fliplr, 27: np.flipud}
    mirrors = [(26,), (27,)]
    if H == W:
        mirrors += [(24,), (25,)]
    mirrors += [(26, 27)]
    if H == W:
        mirrors += [(24, 27), (25, 27)]

    def build(plan, G0):
        """ops overlaying the grid with each mirror in plan; None if a step is idle."""
        G = G0.copy()
        out = []
        for m in plan:
            T = G.copy()
            seq = []
            for op in m:
                nxt = FN[op](T)
                if np.array_equal(nxt, T):
                    return None, None
                seq.append(op)
                T = nxt
            merged = np.where(G != 0, G, T)
            if np.array_equal(merged, G):
                return None, None
            if np.array_equal(merged, T):
                out += [(op, whole) for op in seq]          # mirroring alone restores it
            else:
                out.append((29, whole))                     # copy current grid
                out += [(op, whole) for op in seq]          # mirror the whole grid
                out.append((30, [0, 0, 0, 0]))              # lay the copy back over it
            G = merged
        return out, G

    def replay(G0, seq):
        G = G0.copy()
        clip = np.zeros_like(G)
        for op, _s in seq:
            if op == 29:
                clip = G.copy()
            elif op == 30:
                G = np.where(clip != 0, clip, G)
            else:
                G = FN[op](G)
        return G

    # the fully restored picture = overlay under every mirror (computed from I alone)
    full = G.copy()
    for _ in range(3):
        for m in mirrors:
            T = full.copy()
            for op in m:
                T = FN[op](T)
            full = np.where(full != 0, full, T)

    from itertools import permutations
    cands = []
    for k in range(1, 4):
        for plan in permutations(mirrors, k):
            seq, res = build(plan, G)
            if seq is None or not np.array_equal(res, full):
                continue
            # every op must matter: dropping any one must break the restoration
            if any(np.array_equal(replay(G, seq[:i] + seq[i + 1:]), full)
                   for i in range(len(seq))):
                continue
            cands.append((len(plan), len(seq), seq))
        if cands:
            break
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))
        for op, s in cands[0][2]:
            ops.append(op); sels.append(s)

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b8825c91"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8825c91"
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
                                f"for task b8825c91"
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
                    f"Failed to build a complete episode for task b8825c91 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8825c91-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
