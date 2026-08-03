"""
ARC Task: 6d58a25d (RE-ARC) — LLM-generated grid_maker
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

# The template shape used by this task (normalized, 4 rows x 7 cols) in its
# canonical orientation.  In that orientation the rays always shoot DOWN.
CANON_SHAPE = frozenset({
    (0, 3), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 4), (2, 5), (3, 0), (3, 6)
})

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1 = random.choice(rem)                                  # template shape color
    c2 = random.choice([c for c in rem if c != c1])          # noise / ray color

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, c1=None, c2=None, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    if bgc is None:
        bgc = choice(interval(0, 10, 1))
    if c1 is None:
        c1 = choice(remove(bgc, interval(0, 10, 1)))
    if c2 is None:
        c2 = choice(remove(c1, remove(bgc, interval(0, 10, 1))))

    shp = normalize(frozenset({
        (0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)
    }))

    # a rot90/rot270 instance swaps the final dimensions
    if rot in ("rot90", "rot270"):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    hub = max(5, min(30, hub))
    wub = max(8, min(30, wub))

    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (8, wub))

    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    loci = randint(0, h - 4)
    locj = randint(0, w - 7)
    plcd = shift(shp, (loci, locj))
    rem = difference(inds, plcd)
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = sample(rem, nnoise)
    gi = fill(c, c2, nois)
    gi = fill(gi, c1, plcd)

    ff = lambda ij: len(intersection(shoot(ij, (-1, 0)), plcd)) > 0
    trg = sfilter(nois, ff)
    gg = lambda ij: valmax(sfilter(plcd, lambda kl: kl[1] == ij[1]), first) + 1
    kk = lambda ij: connect((gg(ij), ij[1]), (h - 1, ij[1]))
    fullres = mapply(kk, trg)
    go = fill(gi, c2, fullres)

    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _normalize_cells(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def derive_operations(I, O):
    """
    Rule (derived from I alone):
      * one object is the fixed template shape (a 'V' spanning 4 x 7), drawn in c1;
      * everything else non-background is scattered noise in c2;
      * orient the grid so the template sits in its canonical pose -- rays then run
        straight away from the template's open side;
      * a template column fires iff some noise cell lies in that column on the far
        side of the template's first cell there;
      * a firing column emits ONE solid ray of c2 from just past the template's last
        cell in that column all the way to the grid edge.
    Each ray is one region -> one Color op, ordered along the template.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    flat_idx = np.arange(hi * wi).reshape(hi, wi)

    # locate the template + the orientation that puts it in canonical pose
    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(flat_idx, k)
        for col in sorted(set(J.flatten().tolist())):
            if col == bgc:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) != len(CANON_SHAPE):
                continue
            if _normalize_cells(cells) == CANON_SHAPE:
                found = (J, Jidx, col, cells)
                break
        if found is not None:
            break

    if found is None:
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    J, Jidx, shape_col, shape_cells = found
    hJ, wJ = J.shape

    noise_col = None
    for col in sorted(set(J.flatten().tolist())):
        if col != bgc and col != shape_col:
            noise_col = col
            break
    if noise_col is None:
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    # template cells grouped by canonical column
    by_col = {}
    for (r, c) in shape_cells:
        by_col.setdefault(c, []).append(r)

    rays = []
    for j in sorted(by_col):
        first_r = min(by_col[j])
        last_r = max(by_col[j])
        # fires iff a noise cell in this column can "see" the template above it
        fires = any(J[i, j] == noise_col for i in range(first_r, hJ))
        if not fires:
            continue
        start = last_r + 1
        if start > hJ - 1:
            continue
        cells = [(i, j) for i in range(start, hJ)]
        if all(J[i, jj] == noise_col for i, jj in cells):
            continue  # ray already fully drawn in I
        rays.append(cells)

    # one Color op per ray, walking the template's columns in order
    for cells in rays:
        pts = [divmod(int(Jidx[r, c]), wi) for (r, c) in cells]
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        ops.append(int(noise_col))
        sels.append([r0, c0, r1 - r0, c1 - c0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
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
                                f"for task 6d58a25d"
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
                    f"Failed to build a complete episode for task 6d58a25d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d58a25d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
