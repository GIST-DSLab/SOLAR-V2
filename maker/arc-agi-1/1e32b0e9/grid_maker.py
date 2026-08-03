"""
ARC Task: 1e32b0e9 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (4, max(4, min(6, max_h))))
    w = unifint(diff_lb, diff_ub, (4, max(4, min(6, max_w))))
    nh_cap = max(1, min(4, (max_h + 1) // (h + 1)))
    nw_cap = max(1, min(3, (max_w + 1) // (w + 1)))
    nh = unifint(diff_lb, diff_ub, (1, nh_cap))
    nw_lb = 1 if nh > 1 else 2
    if nw_cap < nw_lb:
        if nh == 1 and nh_cap >= 2:
            nh = 2
            nw_lb = 1
        else:
            nw_lb = nw_cap
    nw = unifint(diff_lb, diff_ub, (nw_lb, max(nw_lb, nw_cap)))

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    remlocs = remove(srcloc, llocs)
    ncells = unifint(diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    smallc2 = canvas(bgc, (h - 2, w - 2))
    inds = asindices(smallc2)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    go = tuple(e for e in gi)
    ncc = ncells + 1
    for rl in remlocs:
        nleft = randint(0, ncc)
        subobj = sample(totuple(shp), nleft)
        sg2 = asobject(fill(smallc, fgc, subobj))
        gi = paint(gi, shift(sg2, rl))
        go = paint(go, shift(gg2, rl))
        go = fill(go, fgc, shift(subobj, rl))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- 1. find the separator colour: colours of fully-constant lines, least common ---
    cand = []
    for r in range(H):
        if len(set(I[r, :].tolist())) == 1:
            cand.append(int(I[r, 0]))
    for c in range(W):
        if len(set(I[:, c].tolist())) == 1:
            cand.append(int(I[0, c]))
    cnt = Counter(cand)
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # --- 2. separator rows/cols -> block bands ---
    srows = set(r for r in range(H) if all(int(v) == linc for v in I[r, :]))
    scols = set(c for c in range(W) if all(int(v) == linc for v in I[:, c]))

    def bands(n, seps):
        res, cur = [], []
        for i in range(n):
            if i in seps:
                if cur:
                    res.append((cur[0], len(cur)))
                    cur = []
            else:
                cur.append(i)
        if cur:
            res.append((cur[0], len(cur)))
        return res

    rbands = bands(H, srows)
    cbands = bands(W, scols)

    # --- 3. background / foreground colours inside blocks ---
    inner = []
    for (r0, bh) in rbands:
        for (c0, bw) in cbands:
            for r in range(r0, r0 + bh):
                for c in range(c0, c0 + bw):
                    inner.append(int(I[r, c]))
    icnt = Counter(inner)
    bgc = icnt.most_common(1)[0][0]
    fg_candidates = [k for k in icnt if k != bgc]
    if not fg_candidates:
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels
    fgc = max(fg_candidates, key=lambda k: icnt[k])

    # --- 4. per-block foreground cell sets, normalized to block origin ---
    blocks = []
    for (r0, bh) in rbands:
        for (c0, bw) in cbands:
            rel = set()
            for r in range(bh):
                for c in range(bw):
                    if int(I[r0 + r, c0 + c]) == fgc:
                        rel.add((r, c))
            blocks.append({"r0": r0, "c0": c0, "rel": rel})

    # --- 5. the source block: the one whose shape contains every other block's cells ---
    src = None
    for b in blocks:
        if all(o["rel"] <= b["rel"] for o in blocks):
            src = b
            break
    if src is None:
        src = max(blocks, key=lambda b: len(b["rel"]))
    shape = src["rel"]

    # --- 6. stamp the source shape into every other block; only the cells the block
    #        is missing become linc (cells already present stay fgc) ---
    for b in blocks:
        if b is src:
            continue
        missing = sorted(shape - b["rel"])
        if not missing:
            continue
        cells = [(b["r0"] + r, b["c0"] + c) for (r, c) in missing]
        ops.append(int(linc))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
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
                                f"for task 1e32b0e9"
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
                    f"Failed to build a complete episode for task 1e32b0e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e32b0e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
