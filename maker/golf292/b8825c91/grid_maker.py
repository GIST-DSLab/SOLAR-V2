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
from itertools import product
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice([c for c in range(10) if c != 4])
    return {"bgc": bgc}


def _unifint(lb, ub, bounds):
    a, b = bounds
    lo = int(round(a + (b - a) * lb))
    hi = int(round(a + (b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    for (i, j), c in obj.items():
        g[i, j] = c
    gt = g.T.copy()
    for i in range(h):
        for j in range(i, w):
            g[i, j] = gt[i, j]
    g = np.hstack([g, np.fliplr(g)])
    g = np.vstack([g, np.flipud(g)])
    gi, go = g.copy(), g.copy()
    for alph in (2, 1):
        locidev = _unifint(diff_lb, diff_ub, (1, alph * h))
        locjdev = _unifint(diff_lb, diff_ub, (1, w))
        loci = alph * h - locidev
        locj = w - locjdev
        loci2 = _unifint(diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = _unifint(diff_lb, diff_ub, (locj, w - 1))
        gi[loci:loci2 + 1, locj:locj2 + 1] = 4
        gi, go = np.rot90(gi, 2), np.rot90(go, 2)
    mfs = [lambda x: x, lambda x: x.T, lambda x: np.rot90(x, 2).T,
           np.fliplr, np.flipud, lambda x: np.rot90(x, 3),
           lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi, go = fn(gi), fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Rule read off I: the picture has full square (D4) symmetry; rectangles of colour 4
    # hide parts of it.  Blank the 4-cells (Colour0 on "all cells of colour 4"), then
    # merge the grid with a mirrored/rotated copy of itself: CopyO the whole grid, flip or
    # rotate the whole grid, Paste the copy back at (0,0).  Paste is transparent to 0, so
    # every intact cell stays and every blanked cell takes its symmetry partner's value.
    # One or two such rounds (chosen from I alone, never from O) fill every hole.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    full = [0, 0, H - 1, W - 1]          # whole square grid: selection is the full rectangle
    TR = {26: np.fliplr, 27: np.flipud,
          24: lambda x: np.rot90(x, 1), 25: lambda x: np.rot90(x, 3)}

    # completed picture as implied by I's own symmetry (used only to choose a plan)
    target = I.copy()
    for f in (np.fliplr, np.flipud, lambda x: np.rot90(x, 2), lambda x: x.T,
              lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 3),
              lambda x: np.rot90(x, 2).T):
        src = f(I)
        m = (target == 4) & (src != 4)
        target[m] = src[m]

    ops, sels = [], []
    four = [(int(r), int(c)) for r, c in zip(*np.nonzero(I == 4))]
    G0 = I.copy()
    G0[I == 4] = 0
    ops.append(0); sels.append(sel_of(four))     # blank every cell of colour 4

    def run(seq):
        G = G0.copy()
        for t in seq:
            T = TR[t](G)
            if np.array_equal(T, G):
                return None
            new = np.where(G != 0, G, T)
            if np.array_equal(new, G):
                return None
            G = new
        return G

    plan = None
    for L in range(0, 4):
        for seq in product(TR.keys(), repeat=L):
            G = run(seq)
            if G is not None and np.array_equal(G, target):
                plan = seq
                break
        if plan is not None:
            break
    for t in plan:
        ops.append(29); sels.append(full)          # copy the whole working grid
        ops.append(t); sels.append(full)           # mirror/rotate the whole grid
        ops.append(30); sels.append([0, 0, 0, 0])  # lay the copy back on top (0s transparent)
    ops.append(34); sels.append(full)
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
