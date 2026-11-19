"""
ARC Task: d07ae81c (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random as _rnd
    c1, c2, c3, c4 = _rnd.sample(list(range(10)), 4)
    # c1 / c2 : the two region colors ; c3 marks c1-regions, c4 marks c2-regions
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             c1: int, c2: int, c3: int, c4: int) -> dict:
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    hlo, wlo = min(10, max_h), min(10, max_w)
    gi = None
    for _attempt in range(30):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        magiccol = 0
        g = canvas(0, (h, w))
        ndivi = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
        for k in range(ndivi):
            objs = objects(g, T, F, F)
            objs = sfilter(objs, lambda o: min(shape(o)) > 3 and max(shape(o)) > 4)
            objs = sfilter(objs, lambda o: height(o) * width(o) == len(o))
            if len(objs) == 0:
                break
            obj = choice(totuple(objs))
            if choice((True, False)):
                loci = randint(uppermost(obj) + 2, lowermost(obj) - 1)
                newobj = backdrop(frozenset({(loci, leftmost(obj)), lrcorner(obj)}))
            else:
                locj = randint(leftmost(obj) + 2, rightmost(obj) - 1)
                newobj = backdrop(frozenset({(uppermost(obj), locj), lrcorner(obj)}))
            magiccol += 1
            g = fill(g, magiccol, newobj)
        objs = objects(g, T, F, F)
        for ii, obj in enumerate(objs):
            col = c1 if ii == 0 else (c2 if ii == 1 else choice((c1, c2)))
            g = fill(g, col, toindices(obj))
        ofc1 = ofcolor(g, c1)
        ofc2 = ofcolor(g, c2)
        if len(ofc1) == 0 or len(ofc2) == 0:
            continue
        mn = min(len(ofc1), len(ofc2))
        n1 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
        n2 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
        srcs1 = set()
        for k in range(n1):
            cands = totuple((ofc1 - srcs1) - mapply(neighbors, srcs1))
            if len(cands) == 0:
                break
            srcs1.add(choice(cands))
        srcs2 = set()
        for k in range(n2):
            cands = totuple((ofc2 - srcs2) - mapply(neighbors, srcs2))
            if len(cands) == 0:
                break
            srcs2.add(choice(cands))
        if len(srcs1) == 0 or len(srcs2) == 0:
            continue
        gi = fill(g, c3, frozenset(srcs1))
        gi = fill(gi, c4, frozenset(srcs2))
        lns = mapply(lnf, frozenset(srcs1)) | mapply(lnf, frozenset(srcs2))
        ofc3 = ofc1 & lns
        ofc4 = ofc2 & lns
        go = fill(gi, c3, ofc3)
        go = fill(go, c4, ofc4)
        return {'input': gi, 'output': go}
    return {'input': gi, 'output': gi}


def derive_operations(I, O, examples=None):
    """Each isolated marker cell shoots four diagonal rays.  A ray is drawn stretch by
    stretch, outward from its marker: the piece lying over region-colour A becomes A's
    marker colour, the piece over region-colour B becomes B's marker colour."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # --- which colours are the markers: a colour whose cells never touch each other ---
    cells_by_col = {}
    for r in range(hi):
        for c in range(wi):
            cells_by_col.setdefault(int(I[r, c]), set()).add((r, c))

    def isolated(cs):
        for (r, c) in cs:
            for dr, dc in NB:
                if (r + dr, c + dc) in cs:
                    return False
        return True

    markers = sorted([col for col, cs in cells_by_col.items() if isolated(cs)])
    if len(markers) != 2:
        markers = sorted([col for col, _ in
                          sorted(cells_by_col.items(), key=lambda kv: len(kv[1]))][:2])
    regions = sorted([col for col in cells_by_col if col not in markers])

    # --- which marker belongs to which region colour (read off the demonstrations) ---
    votes = {m: Counter() for m in markers}
    if examples:
        for pair in examples:
            ei = np.asarray(pair[0], dtype=int)
            eo = np.asarray(pair[1], dtype=int)
            if ei.shape != eo.shape:
                continue
            for r in range(ei.shape[0]):
                for c in range(ei.shape[1]):
                    a, b = int(ei[r, c]), int(eo[r, c])
                    if a != b and b in votes and a in regions:
                        votes[b][a] += 1
    for m in markers:                       # fallback: the region a marker sits inside
        if not votes[m]:
            for (r, c) in cells_by_col.get(m, ()):
                for dr, dc in NB:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hi and 0 <= cc < wi:
                        v = int(I[rr, cc])
                        if v in regions:
                            votes[m][v] += 1
    m2r = {}
    if len(markers) == 2 and len(regions) == 2:
        m0, m1 = markers
        r0, r1 = regions
        if votes[m0][r0] + votes[m1][r1] >= votes[m0][r1] + votes[m1][r0]:
            m2r = {m0: r0, m1: r1}
        else:
            m2r = {m0: r1, m1: r0}
    else:
        for m in markers:
            if votes[m]:
                m2r[m] = votes[m].most_common(1)[0][0]
    r2m = {v: k for k, v in m2r.items()}

    # --- walk every ray outward, cutting it into same-region stretches ---
    raw = []
    marker_cells = sorted([p for m in markers for p in cells_by_col.get(m, ())])
    for (mr, mc) in marker_cells:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            run, run_col = [], None
            r, c = mr + dr, mc + dc
            while 0 <= r < hi and 0 <= c < wi:
                paint = r2m.get(int(I[r, c]))
                if paint is None:                 # crossing another marker: cut the stretch
                    if run:
                        raw.append((run_col, run))
                    run, run_col = [], None
                elif paint == run_col:
                    run.append((r, c))
                else:
                    if run:
                        raw.append((run_col, run))
                    run, run_col = [(r, c)], paint
                r += dr
                c += dc
            if run:
                raw.append((run_col, run))

    # drop stretches an earlier ray has already drawn (no-op at issue time)
    cur = I.copy()
    steps = []
    for col, run in raw:
        if any(int(cur[r, c]) != col for (r, c) in run):
            steps.append((col, run))
            for (r, c) in run:
                cur[r, c] = col
    # drop stretches wholly re-drawn by a later ray (deletable without changing the result)
    keep, covered = [], set()
    for col, run in reversed(steps):
        if any(p not in covered for p in run):
            keep.append((col, run))
        covered.update(run)
    keep.reverse()

    ops, sels = [], []
    for col, run in keep:
        ops.append(int(col))
        sels.append(sel_of(run))              # mask = exactly this diagonal stretch
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])       # full-grid rectangle: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
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
                                f"for task d07ae81c"
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
                    f"Failed to build a complete episode for task d07ae81c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d07ae81c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
