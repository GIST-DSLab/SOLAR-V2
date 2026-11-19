"""
ARC Task: 54d9e175 (RE-ARC) — LLM-generated grid_maker
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
    # Separator colour and cell colour are drawn from {0, 5} and differ.
    # Fix them once for the whole episode so every instance uses the same scheme.
    linc, bgc = random.sample([0, 5], 2)
    return {"linc": linc, "bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, linc=None, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if linc is None or bgc is None:
        linc, bgc = random.sample([0, 5], 2)

    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh_max = max(1, (max_h + 1) // (h + 1))
    nw_max = max(1, (max_w + 1) // (w + 1))
    nh = unifint(diff_lb, diff_ub, (1, nh_max))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, nw_max))
    nh = max(1, min(nh, nh_max))
    nw = max(1, min(nw, nw_max))
    if nh == 1 and nw == 1:
        if nw_max >= 2:
            nw = 2
        elif nh_max >= 2:
            nh = 2
    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1

    gi = [[linc] * fullw for _ in range(fullh)]
    go = [[linc] * fullw for _ in range(fullh)]
    for a in range(nh):
        for b in range(nw):
            r0, c0 = a * (h + 1), b * (w + 1)
            icol = random.randint(1, 4)
            ocol = icol + 5
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    gi[r][c] = bgc
                    go[r][c] = ocol
            dr = random.randrange(h)
            dc = random.randrange(w)
            gi[r0 + dr][c0 + dc] = icol
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    markers = {1, 2, 3, 4}
    n_markers = int(sum(1 for r in range(hi) for c in range(wi) if I[r, c] in markers))

    def comps_for(L):
        seen = np.zeros((hi, wi), dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] == L or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != L:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append(cells)
        return out

    # A cell is a solid rectangle that holds exactly one marker.
    def valid(comps):
        if not comps or len(comps) != n_markers:
            return False
        for cells in comps:
            rs = [y for y, _ in cells]
            cs = [x for _, x in cells]
            if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) != len(cells):
                return False
            if sum(1 for y, x in cells if int(I[y, x]) in markers) != 1:
                return False
        return True

    # Find the separator colour: the one (0 or 5) whose complement splits
    # the grid into rectangular cells with one marker each.
    order = [0, 5]
    if examples:
        # Use the demonstrations to decide which separator colour to try first.
        try:
            from collections import Counter
            votes = Counter()
            for ei, _eo in examples:
                ei = np.asarray(ei, dtype=int)
                for L in (0, 5):
                    if any(all(v == L for v in row) for row in ei.tolist()) or \
                       any(all(v == L for v in col) for col in ei.T.tolist()):
                        votes[L] += 1
            if votes:
                best = votes.most_common(1)[0][0]
                order = [best] + [x for x in order if x != best]
        except Exception:
            pass

    chosen = None
    for L in order:
        cmp_ = comps_for(L)
        if valid(cmp_):
            chosen = cmp_
            break
    if chosen is None:
        from collections import Counter
        L = Counter(I.flatten().tolist()).most_common(1)[0][0]
        chosen = comps_for(L)

    # Order the cells band by band (top to bottom), left to right in each band.
    cells_info = []
    for cells in chosen:
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        marker = next((int(I[y, x]) for y, x in cells if int(I[y, x]) in markers), None)
        if marker is None:
            continue
        cells_info.append((min(rs), min(cs), cells, marker))
    cells_info.sort(key=lambda t: (t[0], t[1]))

    ops, sels = [], []
    for _r0, _c0, cells, marker in cells_info:
        # Fill the whole cell with its marker colour + 5.
        ops.append(marker + 5)
        sels.append(sel_of(sorted(cells)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 54d9e175"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 54d9e175"
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
                                f"for task 54d9e175"
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
                    f"Failed to build a complete episode for task 54d9e175 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"54d9e175-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
