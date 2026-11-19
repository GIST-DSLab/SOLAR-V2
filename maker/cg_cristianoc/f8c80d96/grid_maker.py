"""
ARC Task: f8c80d96 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def _box(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c)); cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0)); cells.add((r, c1))
    return cells


def _clip(cells, h, w):
    return sorted((r, c) for r, c in cells if 0 <= r < h and 0 <= c < w)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 5]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c != 5]
        bgc, linc = random.sample(cols, 2)
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    ow = random.randint(1, 3 if h > 10 else 2)
    oh = random.randint(1, 3 if w > 10 else 2)
    loci = random.randint(-oh + 1, h - 1)
    locj = random.randint(-ow + 1, w - 1)
    ulci, ulcj = loci - 1, locj - 1
    lrci, lrcj = loci + oh, locj + ow
    hoffs = random.randint(2, 4 if h > 12 else 3)
    woffs = random.randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        ln = _clip(_box(ulci - hoffs * k, ulcj - woffs * k,
                        lrci + hoffs * k, lrcj + woffs * k), h, w)
        if ln:
            lns.append(ln)
    nlns = len(lns)
    nmissing = _unifint(diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), 5, dtype=int)
    for k in range(npresent):
        for r, c in lns[k]:
            gi[r, c] = linc
    for ln in lns:
        for r, c in ln:
            go[r, c] = linc
    return {"input": gi.tolist(), "output": go.tolist()}


def _rings(h, w, ulci, ulcj, lrci, lrcj, hoffs, woffs):
    rings = []
    k = 0
    while True:
        ln = _clip(_box(ulci - hoffs * k, ulcj - woffs * k,
                        lrci + hoffs * k, lrcj + woffs * k), h, w)
        if not ln:
            break
        rings.append(ln)
        k += 1
    return rings


def _fit(I, linc):
    # find innermost box + (hoffs, woffs) whose first n rings reproduce exactly the line cells of I
    h, w = I.shape
    L = (I == linc)
    lset = {(int(r), int(c)) for r, c in zip(*np.nonzero(L))}
    for oh in range(1, 4):
        for ow in range(1, 4):
            for ulci in range(-oh, h - 1):
                lrci = ulci + oh + 1
                for ulcj in range(-ow, w - 1):
                    lrcj = ulcj + ow + 1
                    r0 = _clip(_box(ulci, ulcj, lrci, lrcj), h, w)
                    if not r0 or any(not L[r, c] for r, c in r0):
                        continue
                    for hoffs in range(2, 5):
                        for woffs in range(2, 5):
                            rings = _rings(h, w, ulci, ulcj, lrci, lrcj, hoffs, woffs)
                            acc = set()
                            for n, rg in enumerate(rings):
                                acc |= set(rg)
                                if not acc <= lset:
                                    break
                                if acc == lset and n >= 1:
                                    return rings, n + 1
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    counts = {c: int((I == c).sum()) for c in set(I.flatten().tolist())}
    colors = sorted(counts, key=lambda c: counts[c])
    fit = None
    linc = bgc = None
    for cand in colors:
        others = [c for c in colors if c != cand]
        if len(others) != 1:
            continue
        fit = _fit(I, cand)
        if fit is not None:
            linc, bgc = cand, others[0]
            break
    if fit is None:
        raise ValueError("no ring family fits the input")
    rings, npresent = fit

    # 1) continue the nested box family outward: one stamp per missing ring,
    #    starting with the ring just outside the outermost drawn one
    for k in range(npresent, len(rings)):
        ops.append(int(linc)); sels.append(sel_of(rings[k]))

    # 2) whatever is still background in the current grid turns to 5
    G = I.copy()
    for k in range(npresent, len(rings)):
        for r, c in rings[k]:
            G[r, c] = linc
    bg_cells = [(r, c) for r in range(h) for c in range(w) if G[r, c] == bgc]
    if bg_cells:
        ops.append(5); sels.append(sel_of(bg_cells))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
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
                                f"for task f8c80d96"
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
                    f"Failed to build a complete episode for task f8c80d96 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8c80d96-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
