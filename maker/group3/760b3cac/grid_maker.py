"""
ARC Task: 760b3cac (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc, indc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "indc": indc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, indc) -> dict:
    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)
    m = min(30, max_h, max_w)
    h = unifint(diff_lb, diff_ub, (5, m))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(14, (m - 1) // 2))))
    w = 2 * w + 1
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    if choice((True, False)):
        obj = objL
        sgn = -1
    else:
        obj = objR
        sgn = 1
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))
    inds = asindices(c)
    sp = choice(totuple(inds))
    objx = {sp}
    numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for k in range(numc - 1):
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    while width(objx) != objw:
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    objx = normalize(objx)
    objh, objw = shape(objx)
    loci = randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)
    objx2 = vmirror(plcd)
    plcd2 = shift(objx2, (0, objw * sgn))
    go = fill(gi, objc, plcd2)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _is_marker(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    if len(cells) != 5 or max(rs) - r0 != 2 or max(cs) - c0 != 2:
        return False
    if (r0 + 1, c0 + 1) not in cells:
        return False
    corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
    return len(cells & corners) == 1


def _marker_direction(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    dr = dc = 0
    if sum(1 for r in rs if r == max(rs)) == 2:
        dr += 1
    if sum(1 for r in rs if r == min(rs)) == 2:
        dr -= 1
    if sum(1 for c in cs if c == max(cs)) == 2:
        dc += 1
    if sum(1 for c in cs if c == min(cs)) == 2:
        dc -= 1
    if dr != 0 and dc != 0:
        dc = 0
    if dr == 0 and dc == 0:
        dr = 1
    return dr, dc


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: majority colour across this input and the demonstration inputs
    cnt = Counter(I.flatten().tolist())
    for ex_i, _ in (examples or []):
        cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, set()).add((r, c))
    if len(groups) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the object colour is the one the demonstrations grow; otherwise the
    # non-arrow-shaped object
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int)
        ex_o = np.asarray(ex_o, dtype=int)
        if ex_i.shape != ex_o.shape:
            continue
        for v in np.unique(ex_o).tolist():
            if int((ex_o == v).sum()) > int((ex_i == v).sum()):
                grown[int(v)] += 1
    main_col = None
    for v, _ in grown.most_common():
        if v in groups:
            main_col = v
            break
    if main_col is None:
        markers = [k for k in groups if _is_marker(groups[k])]
        rest = [k for k in groups if k not in markers]
        main_col = max(rest or list(groups), key=lambda k: len(groups[k]))
    marker_col = [k for k in groups if k != main_col][0]
    main = groups[main_col]
    dr, dc = _marker_direction(groups[marker_col])

    rs = [r for r, _ in main]
    cs = [c for _, c in main]
    r0, c0 = min(rs), min(cs)
    oh, ow = max(rs) - r0 + 1, max(cs) - c0 + 1

    # Unit of work: one LINE of the object at a time.  Mirroring across the
    # edge facing the marker keeps each line parallel to that edge intact and
    # only reverses their order, so every line is carried across on its own:
    # the line nearest the edge lands right beside it, then outward.
    if dc == 0:
        src_lines = list(range(r0 + oh - 1, r0 - 1, -1)) if dr > 0 else list(range(r0, r0 + oh))

        def strip(k):      # full row strip of the object's bbox (background included, on purpose)
            return [k, c0, 0, ow - 1]

        def dest(k):
            return (2 * r0 + 2 * oh - 1 - k, c0) if dr > 0 else (2 * r0 - 1 - k, c0)

        def content(k):
            return tuple(I[k, c0:c0 + ow].tolist())

        def line_cells(k):
            return [(rr, cc) for (rr, cc) in main if rr == k]

        def mirror_cell(p, k):
            return (dest(k)[0], p[1])
    else:
        src_lines = list(range(c0 + ow - 1, c0 - 1, -1)) if dc > 0 else list(range(c0, c0 + ow))

        def strip(k):      # full column strip of the object's bbox (background included, on purpose)
            return [r0, k, oh - 1, 0]

        def dest(k):
            return (r0, 2 * c0 + 2 * ow - 1 - k) if dc > 0 else (r0, 2 * c0 - 1 - k)

        def content(k):
            return tuple(I[r0:r0 + oh, k].tolist())

        def line_cells(k):
            return [(rr, cc) for (rr, cc) in main if cc == k]

        def mirror_cell(p, k):
            return (p[0], dest(k)[1])

    clip = None
    for k in src_lines:
        d = dest(k)
        if not (0 <= d[0] < hi and 0 <= d[1] < wi):
            continue
        if main_col != 0:
            # copy this input line, drop it on its mirrored line; a line identical
            # to the one already on the clipboard is just pasted again
            if content(k) != clip:
                ops.append(28); sels.append(strip(k))
                clip = content(k)
            ops.append(30); sels.append([d[0], d[1], 0, 0])
        else:
            # colour 0 is transparent to Copy/Paste: draw this mirrored line directly
            tgt = [mirror_cell(p, k) for p in line_cells(k)]
            ops.append(0); sels.append(sel_of(tgt))

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
                        f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
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
                                f"for task 760b3cac"
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
                    f"Failed to build a complete episode for task 760b3cac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"760b3cac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
