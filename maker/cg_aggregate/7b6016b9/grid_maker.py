"""
ARC Task: 7b6016b9 (RE-ARC) — LLM-generated grid_maker
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
    # colours 2 and 3 are reserved for the output roles
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    v = random.uniform(min(a, b), max(a, b))
    return max(lo, min(hi, int(round(v))))


def _bg_components(g, bgc):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                bord = any(r_ in (0, h - 1) or c_ in (0, w - 1) for r_, c_ in cells)
                comps.append((cells, bord))
    return comps


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, **kw) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c not in (2, 3)]
        bgc, fgc = random.sample(cols, 2)
    for _ in range(500):
        h = _unifint(diff_lb, diff_ub, 5, max(5, max_h))
        w = _unifint(diff_lb, diff_ub, 5, max(5, max_w))
        numl = _unifint(diff_lb, diff_ub, 4, min(h, w))
        gi = np.full((h, w), bgc, dtype=int)
        iopts = list(range(1, h - 1))
        jopts = list(range(1, w - 1))
        numlh = random.randint(numl // 3, numl // 3 * 2)
        numlw = numl - numlh
        for _k in range(numlh):
            if not iopts:
                continue
            loci = random.choice(iopts)
            iopts = [x for x in iopts if x not in (loci - 1, loci, loci + 1)]
            a, b = random.sample(range(w), 2)
            a = random.randint(0, a)
            b = random.randint(b, w - 1)
            lo, hi_ = min(a, b), max(a, b)
            gi[loci, lo:hi_ + 1] = fgc
        for _k in range(numlw):
            if not jopts:
                continue
            locj = random.choice(jopts)
            jopts = [x for x in jopts if x not in (locj - 1, locj, locj + 1)]
            a, b = random.sample(range(h), 2)
            a = random.randint(0, a)
            b = random.randint(b, h - 1)
            lo, hi_ = min(a, b), max(a, b)
            gi[lo:hi_ + 1, locj] = fgc
        tofill = set()
        for cells, bord in _bg_components(gi, bgc):
            if not bord:
                tofill.update(cells)
        if not tofill:
            continue
        for r, c in tofill:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in tofill and 0 <= nr < h and 0 <= nc < w:
                        gi[nr, nc] = fgc
        # output computed from final input by the rule (verifier-consistent)
        go = gi.copy()
        any_enc = False
        for cells, bord in _bg_components(gi, bgc):
            for r, c in cells:
                go[r, c] = 3 if bord else 2
            any_enc = any_enc or not bord
        if not any_enc:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant colour on the grid's outer frame (as the verifier reads it)
    frame = list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1])
    bgc = Counter(int(x) for x in frame).most_common(1)[0][0]

    g = I.copy()
    ops, sels = [], []

    # Reading: the "outside" is whatever background the border reaches.
    # Walk the frame; each time a frame cell is still background, flood the
    # outside from it with 3.
    border = [(0, c) for c in range(wi)] + [(r, wi - 1) for r in range(1, hi)] + \
             [(hi - 1, c) for c in range(wi - 2, -1, -1)] + [(r, 0) for r in range(hi - 2, 0, -1)]
    for r, c in border:
        if g[r, c] != bgc:
            continue
        ops.append(13); sels.append([r, c, 0, 0])   # FloodFill3 from frame seed
        stack = [(r, c)]
        g[r, c] = 3
        while stack:
            cr, cc = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < hi and 0 <= nc < wi and g[nr, nc] == bgc:
                    g[nr, nc] = 3
                    stack.append((nr, nc))

    # Whatever background the outside flood could not reach is enclosed: all
    # remaining background-coloured cells become 2 at once.
    rest = [(r, c) for r in range(hi) for c in range(wi) if g[r, c] == bgc]
    if rest:
        ops.append(2); sels.append(sel_of(rest))
        for r, c in rest:
            g[r, c] = 2

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
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
                                f"for task 7b6016b9"
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
                    f"Failed to build a complete episode for task 7b6016b9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7b6016b9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
