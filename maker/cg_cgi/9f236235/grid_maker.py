"""
ARC Task: 9f236235 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    frontcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "frontcol": frontcol}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, frontcol=1, **kw) -> dict:
    from random import choice, sample
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a: b = a
        lo = a + round(lb * (b - a)); hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))
    lim_h = min(31, max_h + 1); lim_w = min(31, max_w + 1)
    numh = unifint(diff_lb, diff_ub, (2, min(14, lim_h // 2)))
    numw = unifint(diff_lb, diff_ub, (2, min(14, lim_w // 2)))
    h = unifint(diff_lb, diff_ub, (1, max(1, lim_h // numh - 1)))
    w = unifint(diff_lb, diff_ub, (1, max(1, lim_w // numw - 1)))
    remcols = [c for c in range(10) if c != frontcol]
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = [(a, b) for a in range(numh) for b in range(numw)]
    locs = set(sample(cands, numcells))
    gi = np.full((h * numh + numh - 1, w * numw + numw - 1), frontcol, dtype=int)
    go = np.full((numh, numw), bgc, dtype=int)
    for (a, b) in cands:
        col = choice(remcols) if (a, b) in locs else bgc
        gi[a * (h + 1):a * (h + 1) + h, b * (w + 1):b * (w + 1) + w] = col
        go[a, b] = col
    go = go[:, ::-1]
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # frontier lines: the rows/columns of I that are a single colour end to end
    hlines = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = int(I[hlines[0], 0])
    hlines = [r for r in hlines if I[r, 0] == fcol]
    vlines = [c for c in range(wi) if all(I[:, c] == fcol)]
    numh, numw = len(hlines) + 1, len(vlines) + 1
    h = (hi - (numh - 1)) // numh
    w = (wi - (numw - 1)) // numw
    keycols = [b * (w + 1) for b in range(numw)]   # one sample column per block column

    # S[a][b] = the sample pixel (top-left) of block (a, b), read straight from I
    S = I[0::h + 1, 0::w + 1][:numh, :numw]

    g = I.copy()
    ops, sels = [], []
    clip = None   # array currently held in ARCLE's clipboard (0 = transparent)

    def pasted(grid, cl, r0, c0):
        out = grid.copy()
        if cl is None:
            return out
        ch, cw = cl.shape
        sub = out[r0:r0 + ch, c0:c0 + cw]
        part = cl[:sub.shape[0], :sub.shape[1]]
        sub[part != 0] = part[part != 0]
        return out

    def same_on(x, y, cells):
        return all(x[p] == y[p] for p in cells)

    def line_step(copy_op, src_box, src_arr, r0, c0, care):
        # care = cells whose final value is read later AND whose block colour is
        # non-zero (0-coloured blocks are painted at the end anyway)
        nonlocal g, clip
        want = pasted(g, src_arr, r0, c0)
        if same_on(want, g, care):
            return                      # this line is already right where it matters
        if clip is None or clip.shape != src_arr.shape or \
                not same_on(pasted(g, clip, r0, c0), want, care):
            ops.append(copy_op); sels.append(src_box)
            clip = src_arr
        ops.append(30); sels.append([r0, c0, 0, 0])
        g = pasted(g, clip, r0, c0)

    # 1. collapse rows: block row a's first pixel row of the INPUT is copied up to row a.
    for a in range(1, numh):
        s = a * (h + 1)
        care = [(a, keycols[b]) for b in range(numw) if S[a, b] != 0]
        line_step(28, [s, 0, 0, wi - 1], I[s:s + 1, :].copy(), a, 0, care)

    # 2. collapse columns: block column b's sample column (rows 0..numh-1 of the
    #    working grid) is copied left to column b.
    for b in range(1, numw):
        s = b * (w + 1)
        care = [(a, b) for a in range(numh) if S[a, b] != 0]
        line_step(29, [0, s, numh - 1, 0], g[:numh, s:s + 1].copy(), 0, b, care)

    # 3. drop everything outside the collapsed numh x numw corner (exact rectangle)
    ops.append(33); sels.append([0, 0, numh - 1, numw - 1])
    g = g[:numh, :numw].copy()

    # 3b. Copy/Paste cannot carry colour 0: paint the blocks whose colour in I is 0
    zero_cells = [(a, b) for a in range(numh) for b in range(numw)
                  if S[a, b] == 0 and g[a, b] != 0]
    if zero_cells:
        ops.append(0); sels.append(sel_of(zero_cells))
        for a, b in zero_cells:
            g[a, b] = 0

    # 4. mirror left<->right over the whole (rectangular) canvas
    if not np.array_equal(g, g[:, ::-1]):
        ops.append(26); sels.append([0, 0, numh - 1, numw - 1])

    # Drop any line copy whose effect never reaches the result (e.g. it only fixed a
    # cell that a later transparent paste leaves alone anyway). Checked by replaying
    # the ops on I in plain Python -- no reference to O.
    def replay(oo, ss):
        G = I.copy(); cl = None
        for o, sl in zip(oo, ss):
            if o == 28 or o == 29:
                r, c, dh, dw = sl
                src = I if o == 28 else G
                cl = src[r:r + dh + 1, c:c + dw + 1].copy()
            elif o == 30:
                if cl is None or sl[0] + cl.shape[0] > G.shape[0] \
                        or sl[1] + cl.shape[1] > G.shape[1]:
                    return None          # a paste spilling off the canvas: not allowed
                G = pasted(G, cl, sl[0], sl[1])
            elif o == 33:
                G = G[:sl[2] + 1, :sl[3] + 1].copy()
            elif o == 0:
                for r, c in sl["cells"]:
                    G[r, c] = 0
            elif o == 26:
                G = G[:, ::-1].copy()
        return G
    target = replay(ops, sels)
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            if ops[k] not in (28, 29, 30):
                continue
            o2 = ops[:k] + ops[k + 1:]; s2 = sels[:k] + sels[k + 1:]
            red = replay(o2, s2)
            if red is not None and np.array_equal(red, target):
                ops, sels = o2, s2
                changed = True
                break

    ops.append(34); sels.append([0, 0, numh - 1, numw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9f236235"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9f236235"
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
                                f"for task 9f236235"
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
                    f"Failed to build a complete episode for task 9f236235 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9f236235-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
