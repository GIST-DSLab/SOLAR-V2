"""
ARC Task: d9fac9be (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, ringc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "ringc": ringc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ringc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    nnoise1 = unifint(diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    noise1 = random.sample(inds, nnoise1)
    s1 = set(noise1)
    rest = [x for x in inds if x not in s1]
    noise2 = random.sample(rest, min(nnoise2, len(rest)))
    for (i, j) in noise1:
        gi[i][j] = noisec
    for (i, j) in noise2:
        gi[i][j] = ringc
    rng = [(a, b) for a in range(3) for b in range(3) if (a, b) != (1, 1)]

    def occ(col):
        res = []
        for i in range(h - 2):
            for j in range(w - 2):
                if all(gi[i + a][j + b] == col for a, b in rng):
                    res.append((i, j))
        return res

    fp1 = occ(noisec)
    fp2 = occ(ringc)
    for (i, j) in fp1:
        a, b = random.choice(rng)
        gi[i + a][j + b] = random.choice((bgc, ringc))
    for (i, j) in fp2:
        a, b = random.choice(rng)
        gi[i + a][j + b] = random.choice((bgc, noisec))
    loci = random.randint(0, h - 3)
    locj = random.randint(0, w - 3)
    for a, b in rng:
        gi[loci + a][locj + b] = ringc
    gi[loci + 1][locj + 1] = noisec
    return {"input": gi, "output": [[noisec]]}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def ring_windows(G, bg):
        G = np.asarray(G, dtype=int)
        h, w = G.shape
        out = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                ctr = int(G[r, c])
                ring = {int(G[r + dr, c + dc]) for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}
                if len(ring) != 1:
                    continue
                rc = next(iter(ring))
                if bg is not None and (rc == bg or ctr == bg):
                    continue
                if rc == ctr:
                    continue
                out.append((r, c, rc, ctr))
        return out

    # Episode conventions from demonstrations: background, ring colour, centre colour.
    allcols = Counter(I.flatten().tolist())
    ex_ring, ex_ctr = None, None
    if examples:
        for ei, eo in examples:
            allcols.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = allcols.most_common(1)[0][0]
    if examples:
        rc_counter = Counter()
        ct_counter = Counter()
        for ei, eo in examples:
            target = int(np.asarray(eo, dtype=int)[0, 0])
            for (r, c, rc, ctr) in ring_windows(ei, bgc):
                if ctr == target:
                    rc_counter[rc] += 1
                    ct_counter[ctr] += 1
        if rc_counter:
            ex_ring = rc_counter.most_common(1)[0][0]
            ex_ctr = ct_counter.most_common(1)[0][0]

    cands = ring_windows(I, bgc)
    pick = None
    for cand in cands:
        if (ex_ring is None or cand[2] == ex_ring) and (ex_ctr is None or cand[3] == ex_ctr):
            pick = cand
            break
    if pick is None and cands:
        pick = cands[0]
    if pick is None:
        pick = (1, 1, None, int(I[1, 1]))
    cr, cc, _, ctr = pick

    ops, sels = [], []
    # Two-stage zoom: first crop to a named region containing the ringed pixel
    # (the full 3x3 window, else its row band, else its column band), then crop
    # that region down to the centre pixel. Pick the stage whose first crop is
    # actually needed (the second crop alone would land on a different colour).
    options = [
        ([cr - 1, cc - 1, 2, 2], [1, 1, 0, 0], int(I[1, 1])),          # 3x3 ring window (full rectangle)
        ([cr, 0, 0, wi - 1], [0, cc, 0, 0], int(I[0, cc])),            # whole row of the centre
        ([0, cc, hi - 1, 0], [cr, 0, 0, 0], int(I[cr, 0])),            # whole column of the centre
    ]
    chosen = None
    for first, second, if_skipped in options:
        if if_skipped != ctr:
            chosen = (first, second)
            break
    if chosen is not None:
        ops.append(33); sels.append(chosen[0])
        ops.append(33); sels.append(chosen[1])
    else:
        ops.append(33); sels.append([cr, cc, 0, 0])
    ops.append(34); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
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
                                f"for task d9fac9be"
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
                    f"Failed to build a complete episode for task d9fac9be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d9fac9be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
