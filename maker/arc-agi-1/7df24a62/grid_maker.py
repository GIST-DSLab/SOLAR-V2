"""
ARC Task: 7df24a62 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # bgc / noisec / sqc are the three colours the generator samples.
    # sqc and noisec are kept non-zero so the key square is a fully opaque
    # stamp (Copy/Paste treats 0 as "nothing").
    sqc = random.choice([c for c in range(1, 10)])
    noisec = random.choice([c for c in range(1, 10) if c != sqc])
    bgc = random.choice([c for c in range(0, 10) if c not in (sqc, noisec)])
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    # grid gets trimmed by one ring at the end -> visible size is (h-2, w-2)
    hub = max(12, min(32, max_h + 2))
    wub = max(12, min(32, max_w + 2))
    h = unifint(diff_lb, diff_ub, (12, hub))
    w = unifint(diff_lb, diff_ub, (12, wub))
    # square key block (oh == ow) so every D4 orientation of the stamp is a
    # square selection in ARCLE
    oh = unifint(diff_lb, diff_ub, (3, max(3, min(7, h // 3, w // 3))))
    ow = oh

    interior = [(i, j) for i in range(1, oh - 1) for j in range(1, ow - 1)]
    obj = {random.choice(interior)}
    while True:
        rs = [p[0] for p in obj]
        cs = [p[1] for p in obj]
        if (max(rs) - min(rs) + 1 == oh - 2) and (max(cs) - min(cs) + 1 == ow - 2):
            break
        rem = [p for p in interior if p not in obj]
        obj.add(random.choice(rem))

    pat = np.full((oh, ow), sqc, dtype=int)
    for (i, j) in obj:
        pat[i, j] = noisec
    targ = np.where(pat == sqc, bgc, pat)

    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    gi[loci:loci + oh, locj:locj + ow] = pat

    sq_cells = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    forb = set(sq_cells)
    for (i, j) in sq_cells:
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            forb.add((i + di, j + dj))
    # noise (and hence every pattern occurrence) stays two cells away from the
    # untrimmed border, so no stamp gets clipped by the final trim
    inds = {(i, j) for i in range(2, h - 2) for j in range(2, w - 2)} - forb

    if len(inds) > 0:
        namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
        noise = random.sample(sorted(inds), min(namt, len(inds)))
        for (i, j) in noise:
            gi[i, j] = noisec

    D4 = [
        lambda a: a,
        lambda a: np.flipud(a),
        lambda a: np.fliplr(a),
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a, 3),
        lambda a: np.rot90(a, 1),
        lambda a: np.fliplr(np.rot90(a, 3)),
        lambda a: np.flipud(np.rot90(a, 3)),
    ]
    variants = [(np.array(f(targ)), np.array(f(pat))) for f in D4]

    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
    succ, tr, maxtr = 0, 0, 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        t, s = variants[random.randrange(len(variants))]
        hh, ww = t.shape
        cands = [ij for ij in sorted(inds)
                 if 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1]
        if len(cands) == 0:
            continue
        loc = cands[random.randrange(len(cands))]
        tpi = {(loc[0] + i, loc[1] + j) for i in range(hh) for j in range(ww)}
        if tpi <= inds:
            succ += 1
            inds -= tpi
            gi[loc[0]:loc[0] + hh, loc[1]:loc[1] + ww] = t

    # output: wherever the noise pattern occurs (in any orientation) on plain
    # background, the corresponding oriented key square is stamped over it
    go = gi.copy()
    for (t, s) in variants:
        hh, ww = t.shape
        for r in range(h - hh + 1):
            for c in range(w - ww + 1):
                if np.array_equal(gi[r:r + hh, c:c + ww], t):
                    go[r:r + hh, c:c + ww] = s

    gi_t = gi[1:-1, 1:-1]
    go_t = go[1:-1, 1:-1]
    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi_t),
        "output": tuple(tuple(int(v) for v in row) for row in go_t),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    palette = sorted(set(I.flatten().tolist()))

    # 1. locate the key square in I: the most compact colour whose bbox is a
    #    square with a solid ring of that colour and exactly one other colour
    #    inside (that other colour is the noise colour).
    best = None
    for c in palette:
        pos = np.argwhere(I == c)
        r0, c0 = int(pos[:, 0].min()), int(pos[:, 1].min())
        r1, c1 = int(pos[:, 0].max()), int(pos[:, 1].max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh != bw or bh < 3:
            continue
        blk = I[r0:r1 + 1, c0:c1 + 1]
        if not (np.all(blk[0] == c) and np.all(blk[-1] == c)
                and np.all(blk[:, 0] == c) and np.all(blk[:, -1] == c)):
            continue
        inner = set(blk.flatten().tolist()) - {c}
        if len(inner) != 1:
            continue
        if best is None or bh < best[0]:
            best = (bh, int(c), r0, c0, int(list(inner)[0]))

    if best is None:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    oh, sqc, sr, sc, noisec = best
    pat = I[sr:sr + oh, sc:sc + oh]
    rest = [c for c in palette if c != sqc and c != noisec]
    bgc = max(rest, key=lambda c: int((I == c).sum())) if rest else 0

    # 2. every orientation of the key square, expressed as the exact ARCLE ops
    #    that produce it from a plain paste of the square.
    variants = [
        (pat, []),
        (np.flipud(pat), [27]),
        (np.fliplr(pat), [26]),
        (np.rot90(pat, 2), [26, 27]),
        (np.rot90(pat, 3), [25]),
        (np.rot90(pat, 1), [24]),
        (np.fliplr(np.rot90(pat, 3)), [25, 26]),
        (np.flipud(np.rot90(pat, 3)), [25, 27]),
    ]

    # 3. find every place where that orientation's noise pattern sits on bare
    #    background -- those are the spots that receive a stamp.
    stamps = {}
    for (sv, tops) in variants:
        tv = np.where(sv == sqc, bgc, sv)
        for r in range(hi - oh + 1):
            for c in range(wi - oh + 1):
                if (r, c) in stamps:
                    continue
                if np.array_equal(I[r:r + oh, c:c + oh], tv):
                    stamps[(r, c)] = tops

    # 4. copy the key square once, then stamp it (whole object at a time)
    #    on each match and turn it into the matching orientation.
    if stamps:
        ops.append(28)
        sels.append([sr, sc, oh - 1, oh - 1])
        for (r, c) in sorted(stamps):
            ops.append(30)
            sels.append([r, c, 0, 0])
            for o in stamps[(r, c)]:
                ops.append(o)
                sels.append([r, c, oh - 1, oh - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
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
                                f"for task 7df24a62"
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
                    f"Failed to build a complete episode for task 7df24a62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7df24a62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
