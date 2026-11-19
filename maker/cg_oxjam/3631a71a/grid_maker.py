"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    patchcol = random.randrange(10)
    bgc = random.choice([c for c in range(10) if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def _orbits(H, W):
    # symmetry orbits of the cropped (H x W) view of the (H+2) x (W+2) symmetric canvas
    R, C = H + 2, W + 2
    owner = -np.ones((H, W), dtype=int)
    orbits = []
    for r in range(H):
        for c in range(W):
            if owner[r, c] >= 0:
                continue
            s = {(r, c)}
            st = [(r, c)]
            while st:
                a, b = st.pop()
                cands = [(R - 1 - a, b), (a, C - 1 - b)]
                if R == C:
                    cands.append((b, a))
                for x, y in cands:
                    if 0 <= x < R and 0 <= y < C and (x, y) not in s:
                        s.add((x, y))
                        st.append((x, y))
            cells = sorted((a, b) for a, b in s if a < H and b < W)
            for a, b in cells:
                owner[a, b] = len(orbits)
            orbits.append(cells)
    return orbits


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, patchcol=9) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))

    hmax = max(6, min(15, (min(max_h, max_w) + 2) // 2))
    h = unifint(6, hmax)
    w = h
    remcols = [c for c in range(10) if c not in (bgc, patchcol)]
    c = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w) if i >= j]
    ccols = random.sample(remcols, unifint(1, 8))
    cells = random.sample(inds, unifint(1, len(inds)))
    obj = [(random.choice(ccols), ij) for ij in cells]
    for v, (i, j) in obj:
        c[i, j] = v
    c = c.T.copy()
    for v, (i, j) in obj:
        c[i, j] = v
    c = np.hstack([c, c[:, ::-1]])
    c = np.vstack([c, c[::-1, :]])
    go = c[:-2, :-2].copy()
    gi = go.copy()
    H, W = go.shape
    orbits = _orbits(H, W)
    ndist = unifint(1, int((2 * h * 2 * w) ** 0.5))
    succ = tr = 0
    while succ < ndist and tr < 10 * ndist:
        tr += 1
        oh = random.randint(2, h // 2 + 1)
        ow = random.randint(2, w // 2 + 1)
        li = random.randint(0, H - oh)
        lj = random.randint(0, W - ow)
        g2 = gi.copy()
        g2[li:li + oh, lj:lj + ow] = patchcol
        if (g2[:2, :2] == patchcol).any():
            continue
        if any(all(g2[a, b] == patchcol for a, b in orb) for orb in orbits):
            continue
        gi = g2
        succ += 1
    if succ == 0:
        raise ValueError("no patch placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading taken from the golf entry, which splits the work differently from the
    maker. The maker fills each covered cell from its whole symmetry orbit in one
    step. Here the picture has two basic symmetries: the row mirror c -> W+1-c and
    the transpose (r,c) -> (c,r). A covered cell is uncovered by looking at its
    mirror partner, or at its transpose partner. Passes go mirror, then transpose,
    then mirror again, and so on until no cover is left. Cells restored in one
    pass can be the source for later passes.
    Each op covers one cover rectangle-region, in one pass, with one colour: the
    covered cells whose partner, in the grid as it stands now, is not covered.
    That set is built from the current grid only.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    orbits = _orbits(H, W)
    # cover colour: the colour whose removal leaves every orbit single-valued and determined
    best_p, best = None, -1
    for p in sorted(set(I.flatten().tolist())):
        ok, n = True, 0
        for cells in orbits:
            vals = {int(I[a, b]) for a, b in cells if I[a, b] != p}
            npc = sum(1 for a, b in cells if I[a, b] == p)
            if len(vals) > 1 or (not vals and npc):
                ok = False
                break
            n += npc
        if ok and n > best:
            best_p, best = p, n
    if best_p is None or best <= 0:
        return [34], [[0, 0, ho - 1, wo - 1]]
    pc = best_p

    # connected cover regions in I
    isp = (I == pc)
    vis = np.zeros_like(isp)
    regions = []
    for r in range(H):
        for c in range(W):
            if isp[r, c] and not vis[r, c]:
                comp = []
                q = deque([(r, c)])
                vis[r, c] = True
                while q:
                    a, b = q.popleft()
                    comp.append((a, b))
                    for x, y in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                        if 0 <= x < H and 0 <= y < W and isp[x, y] and not vis[x, y]:
                            vis[x, y] = True
                            q.append((x, y))
                regions.append(sorted(comp))

    syms = [lambda a, b: (a, W + 1 - b),   # row mirror of the doubled canvas
            lambda a, b: (b, a)]           # transpose
    G = I.copy()
    for _ in range(4 * (H + W)):
        if not (G == pc).any():
            break
        progress = False
        for T in syms:
            for comp in regions:
                bycol = {}
                for a, b in comp:
                    if G[a, b] != pc:
                        continue
                    x, y = T(a, b)
                    if 0 <= x < H and 0 <= y < W and G[x, y] != pc:
                        bycol.setdefault(int(G[x, y]), []).append((a, b))
                for col in sorted(bycol):
                    cells = sorted(bycol[col])
                    ops.append(col)
                    sels.append(sel_of(cells))
                    for a, b in cells:
                        G[a, b] = col
                    progress = True
        if not progress:
            break

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
                        f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
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
                                f"for task 3631a71a"
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
                    f"Failed to build a complete episode for task 3631a71a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3631a71a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
