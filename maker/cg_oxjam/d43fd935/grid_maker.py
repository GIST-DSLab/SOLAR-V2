"""
ARC Task: d43fd935 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Fix the background and the box colour for the whole episode.
    # The ray colours (dcols) don't need fixing: each ray keeps the colour of
    # whatever pixel it starts from.
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "ccol": ccol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    remcols = [c for c in range(10) if c != bgc and c != ccol]
    ndcols = unifint(diff_lb, diff_ub, (1, 8))
    dcols = random.sample(remcols, ndcols)

    gi = [[bgc] * w for _ in range(h)]
    bd = set()
    for i in range(loci, loci + boxh):
        for j in range(locj, locj + boxw):
            gi[i][j] = ccol
            bd.add((i, j))
    reminds = [(i, j) for i in range(h) for j in range(w) if (i, j) not in bd]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, nnoise)
    inr = lambda i: loci <= i <= loci + boxh - 1
    inc = lambda j: locj <= j <= locj + boxw - 1
    truenoise = [ij for ij in noise if not inr(ij[0]) and not inc(ij[1])]
    rem = [ij for ij in noise if inr(ij[0]) or inc(ij[1])]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]
    for (i, j) in truenoise:
        gi[i][j] = random.choice(dcols)
    go = [row[:] for row in gi]

    for jj in sorted(set(j for _, j in top)):
        col = random.choice(dcols)
        subs = [ij for ij in top if ij[1] == jj]
        for (i, j) in subs:
            gi[i][j] = col
        for i in range(min(i for i, _ in subs), loci):
            go[i][jj] = col
    for jj in sorted(set(j for _, j in bottom)):
        col = random.choice(dcols)
        subs = [ij for ij in bottom if ij[1] == jj]
        for (i, j) in subs:
            gi[i][j] = col
        for i in range(loci + boxh, max(i for i, _ in subs) + 1):
            go[i][jj] = col
    for ii in sorted(set(i for i, _ in left)):
        col = random.choice(dcols)
        subs = [ij for ij in left if ij[0] == ii]
        for (i, j) in subs:
            gi[i][j] = col
        for j in range(min(j for _, j in subs), locj):
            go[ii][j] = col
    for ii in sorted(set(i for i, _ in right)):
        col = random.choice(dcols)
        subs = [ij for ij in right if ij[0] == ii]
        for (i, j) in subs:
            gi[i][j] = col
        for j in range(locj + boxw, max(j for _, j in subs) + 1):
            go[ii][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = np.asarray(O).shape

    def bg_of(G):
        return Counter(G.flatten().tolist()).most_common(1)[0][0]

    def rect_colors(G):
        # Colours whose cells together fill one solid rectangle of at least 2x2
        # (the same idea as the verifier's partition step).
        bg = bg_of(G)
        out = {}
        for col in set(G.flatten().tolist()) - {bg}:
            rs, cs = np.nonzero(G == col)
            r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
            bh, bw = r1 - r0 + 1, c1 - c0 + 1
            if bh >= 2 and bw >= 2 and bh * bw == len(rs):
                out[col] = (int(r0), int(c0), int(bh), int(bw))
        return out

    # The box colour stays the same for the whole episode: read it from the examples.
    learned = None
    if examples:
        votes = Counter()
        for ex_in, _ in examples:
            rc = rect_colors(np.asarray(ex_in, dtype=int))
            if rc:
                votes[max(rc, key=lambda k: rc[k][2] * rc[k][3])] += 1
        if votes:
            learned = votes.most_common(1)[0][0]

    bgc = bg_of(I)
    rc = rect_colors(I)
    ops, sels = [], []
    if rc:
        key = learned if learned in rc else max(rc, key=lambda k: rc[k][2] * rc[k][3])
        br, bc, bh, bw = rc[key]
        G = I.copy()

        # Each row/column that meets the box is swept from its far end toward
        # the box. A pixel carries its colour over the background run in front
        # of it, until the next pixel takes over or the run reaches the box.
        # One Color op per background run.
        def sweep(line):  # cells ordered from the far end to the cell next to the box
            carry = None
            run = []
            for (r, c) in line:
                v = int(G[r, c])
                if v == bgc:
                    if carry is not None:
                        run.append((r, c))
                else:
                    if carry is not None and run:
                        ops.append(carry); sels.append(sel_of(run))
                        for cell in run:
                            G[cell] = carry
                    carry, run = v, []
            if carry is not None and run:
                ops.append(carry); sels.append(sel_of(run))
                for cell in run:
                    G[cell] = carry

        for j in range(bc, bc + bw):                       # columns above the box
            sweep([(r, j) for r in range(0, br)])
        for j in range(bc, bc + bw):                       # columns below the box
            sweep([(r, j) for r in range(hi - 1, br + bh - 1, -1)])
        for i in range(br, br + bh):                       # rows left of the box
            sweep([(i, c) for c in range(0, bc)])
        for i in range(br, br + bh):                       # rows right of the box
            sweep([(i, c) for c in range(wi - 1, bc + bw - 1, -1)])

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
                        f"num_examples+1 ({num_examples + 1}) for task d43fd935"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d43fd935"
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
                                f"for task d43fd935"
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
                    f"Failed to build a complete episode for task d43fd935 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d43fd935-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
