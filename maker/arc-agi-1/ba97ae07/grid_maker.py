"""
ARC Task: ba97ae07 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- 1. colors
VARIANTS = [
    {"mirrored": False},   # under-band is horizontal
    {"mirrored": True},    # under-band is vertical (dmirror applied)
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    acol = random.choice(remcols)                                  # under band
    bcol = random.choice([c for c in remcols if c != acol])         # over band

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice(VARIANTS)["mirrored"]

    h = _unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    if mirrored:
        # after dmirror the grid is transposed; keep result within max dims
        while h > max_w or w > max_h:
            h = _unifint(diff_lb, diff_ub, (3, max(3, min(max_h, max_w))))
            w = _unifint(diff_lb, diff_ub, (3, max(3, min(max_h, max_w))))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]

    lineh = _unifint(diff_lb, diff_ub, (1, max(1, h // 3)))
    linew = _unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
    loci = random.randint(1, h - lineh - 1)
    locj = random.randint(1, w - linew - 1)

    # input: acol band first, bcol band painted on top
    for a in range(lineh):
        for j in range(w):
            gi[loci + a][j] = acol
    for b in range(linew):
        for i in range(h):
            gi[i][locj + b] = bcol

    # output: bcol band first, acol band painted on top
    for b in range(linew):
        for i in range(h):
            go[i][locj + b] = bcol
    for a in range(lineh):
        for j in range(w):
            go[loci + a][j] = acol

    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


# ---------------------------------------------------------------- 3. ops
def _components(mask):
    hh, ww = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for r in range(hh):
        for c in range(ww):
            if mask[r, c] and not seen[r, c]:
                n += 1
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hh and 0 <= nx < ww and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
    return n


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # background = the colour the canvas was painted with (strictly the majority here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # two bands cross; the one drawn UNDERNEATH is cut in two by the other,
    # so it is the colour whose cells form more connected components.
    fg = [c for c in sorted(set(I.flatten().tolist())) if c != bgc]
    best, best_n = None, -1
    for c in fg:
        n = _components(I == c)
        if n > best_n:
            best, best_n = c, n

    if best is not None:
        # the under-band's full extent = bounding box of its (severed) pieces
        rs, cs = np.nonzero(I == best)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        # sanity: this rule must reproduce O; otherwise pick the other colour
        pred = I.copy()
        pred[r0:r1 + 1, c0:c1 + 1] = best
        if not np.array_equal(pred, O):
            for c in fg:
                if c == best:
                    continue
                rs2, cs2 = np.nonzero(I == c)
                a0, a1 = int(rs2.min()), int(rs2.max())
                b0, b1 = int(cs2.min()), int(cs2.max())
                alt = I.copy()
                alt[a0:a1 + 1, b0:b1 + 1] = c
                if np.array_equal(alt, O):
                    best, r0, r1, c0, c1 = c, a0, a1, b0, b1
                    break

        # redraw the whole under-band on top of the crossing band
        band = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        ops.append(int(best))
        sels.append(sel_of(band))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task ba97ae07"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ba97ae07"
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
                                f"for task ba97ae07"
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
                    f"Failed to build a complete episode for task ba97ae07 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ba97ae07-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
