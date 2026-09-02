"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    # generator: cols = {0..9} \ {1,2,4};  bgc = choice(cols)
    # object colors are irrelevant to the rule (every non-bg rectangle is
    # redrawn as 2 / 4 / 1), so only the background needs fixing.
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, 4, box(bd))
            go = fill(go, 1, corners(bd))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. derive
def _components(G):
    """4-connected, single-colour components: list of (color, cells)."""
    h, w = G.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(G[r, c])
            cells = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                rr, cc = dq.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(G[nr, nc]) == col:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            comps.append((col, cells))
    return comps


def _is_full_rect(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    hh = max(rs) - min(rs) + 1
    ww = max(cs) - min(cs) + 1
    return len(cells) == hh * ww


def _bg_of(G):
    """Background = colour of a component that is NOT a perfect rectangle.

    Every planted shape is a solid rectangle by construction, so the only
    non-rectangular components are background ones (this is exactly what the
    task's own specification extracts).  Fallback: majority colour.
    """
    comps = _components(G)
    best = None
    for col, cells in comps:
        if not _is_full_rect(cells):
            if best is None or len(cells) > len(best[1]):
                best = (col, cells)
    if best is not None:
        return int(best[0])
    return int(Counter(G.flatten().tolist()).most_common(1)[0][0])


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background colour: measured from I; demonstrations (same episode,
    #     same background) are used only to break a degenerate tie. ---
    bgc = _bg_of(I)
    if examples:
        votes = []
        for pair in examples:
            try:
                ex_in = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            votes.append(_bg_of(ex_in))
        if votes:
            consensus = Counter(votes).most_common(1)[0][0]
            # trust the consensus when it is actually present in I and I's own
            # reading was a mere majority-colour fallback
            present = set(I.flatten().tolist())
            if consensus in present and bgc not in present:
                bgc = int(consensus)

    # --- the rectangles: every non-background component ---
    rects = []
    for col, cells in _components(I):
        if col == bgc:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        h = max(rs) - r0 + 1
        w = max(cs) - c0 + 1
        if len(cells) != h * w:
            continue  # not a planted rectangle
        rects.append((r0, c0, h, w))
    rects.sort()

    ops, sels = [], []

    for (r0, c0, h, w) in rects:
        body = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        ring = [(r, c) for (r, c) in body
                if r == r0 or r == r0 + h - 1 or c == c0 or c == c0 + w - 1]
        cnrs = [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)]

        # 1) repaint the whole rectangle 2 (base layer)
        ops.append(2)
        sels.append(sel_of(body))
        # 2) draw its border in 4
        if ring:
            ops.append(4)
            sels.append(sel_of(ring))
        # 3) mark its corners with 1
        ops.append(1)
        sels.append(sel_of(cnrs))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
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
                                f"for task b6afb2da"
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
                    f"Failed to build a complete episode for task b6afb2da "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
