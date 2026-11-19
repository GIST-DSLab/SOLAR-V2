"""
ARC Task: b190f7f5 (RE-ARC) — LLM-generated grid_maker
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

# How this differs from the maker: the maker treats each target cell as one unit
# and stamps the source pattern into its block. The code-golf entry reads every
# output cell as  stencil[u % h][v % w] * target[u // h][v // w].
# That reading is two whole-grid steps. First the colour half is upscaled: each
# non-background cell becomes a solid h x w block. Then the two-colour half is
# tiled over it as a stencil, and every cell under the stencil's background goes
# back to background. Both readings agree on the split, the roles and bgc.
# Only the order and grouping of the paint ops change.


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def _split_and_roles(I):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cands = []
    if hi % 2 == 0:
        cands.append([I[:hi // 2, :], I[hi // 2:, :]])
    if wi % 2 == 0:
        cands.append([I[:, :wi // 2], I[:, wi // 2:]])

    def ncol(g):
        return len(set(g.flatten().tolist()))

    def score(hv):
        a, b = hv
        s = 0
        s += 2 in (ncol(a), ncol(b))
        s += a.shape == b.shape
        s += len(set(a.flatten().tolist()) & set(b.flatten().tolist())) == 1
        s += not (min(a.shape) == 1 or min(b.shape) == 1)
        return s

    best = max(cands, key=score)  # verifier scoring; vsplit preferred on ties
    a, b = best
    src, trg = (a, b) if ncol(a) <= ncol(b) else (b, a)
    common = set(src.flatten().tolist()) & set(trg.flatten().tolist())
    bgc = int(sorted(common)[0])
    return src, trg, bgc


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        return random.randint(a + int(round((b - a) * lb)), a + int(round((b - a) * ub)))

    hmax = max(2, min(5, int(max_h ** 0.5), max_h // 2))
    wmax = max(2, min(5, int(max_w ** 0.5), max_w // 2))
    cols = [c for c in range(10) if c != bgc]
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (2, hmax))
        w = unifint(diff_lb, diff_ub, (2, wmax))
        inds = [(i, j) for i in range(h) for j in range(w)]
        numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        numc = random.choice((numcd, h * w - numcd))
        numc = min(max(1, numc), h * w - 1)
        numcd2 = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        numc2 = random.choice((numcd2, h * w - numcd2))
        numc2 = min(max(2, numc2), h * w - 1)
        srclocs = random.sample(inds, numc)
        srccol = random.choice(cols)
        remcols = [c for c in cols if c != srccol]
        numcols = unifint(diff_lb, diff_ub, (2, 8))
        trglocs = random.sample(inds, numc2)
        ccols = random.sample(remcols, numcols)
        fixc1 = random.choice(ccols)
        tcols = [fixc1, random.choice([c for c in ccols if c != fixc1])] + \
                [random.choice(ccols) for _ in trglocs[2:]]
        gisrc = np.full((h, w), bgc, dtype=int)
        for (i, j) in srclocs:
            gisrc[i, j] = srccol
        gitrg = np.full((h, w), bgc, dtype=int)
        for (i, j), c in zip(trglocs, tcols):
            gitrg[i, j] = c
        ordd = [gisrc, gitrg] if random.random() < 0.5 else [gitrg, gisrc]
        gi = np.hstack(ordd) if random.random() < 0.5 else np.vstack(ordd)
        go = np.full((h * h, w * w), bgc, dtype=int)
        for (a, b) in trglocs:
            for (i, j) in srclocs:
                go[a * h + i, b * w + j] = gitrg[a, b]
        # keep only instances where the split, roles and background can be read from the input
        src, trg, bg2 = _split_and_roles(gi)
        if bg2 != bgc or src.shape != (h, w) or not np.array_equal(src, gisrc) \
                or not np.array_equal(trg, gitrg):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("failed to generate")


def derive_operations(I, O=None, examples=None):
    # The output is the colour half upscaled (each cell -> an h x w block),
    # with the two-colour half tiled over it as a stencil. Wherever the stencil
    # is background, the block is cut back to background.
    I = np.asarray(I, dtype=int)
    src, trg, bgc = _split_and_roles(I)
    h, w = src.shape
    H, W = h * h, w * w
    ops, sels = [], []

    # canvas at output size; lay background over the whole canvas
    # (this also clears the input copy the resize leaves in the top-left)
    ops.append(33); sels.append([0, 0, H - 1, W - 1])
    ops.append(bgc); sels.append([0, 0, H - 1, W - 1])

    # upscale: each non-background target cell becomes a solid h x w block
    # (the bbox is exactly the intended full rectangle)
    painted = []
    for r in range(h):
        for c in range(w):
            col = int(trg[r, c])
            if col == bgc:
                continue
            ops.append(col); sels.append([r * h, c * w, h - 1, w - 1])
            painted.append((r, c))

    # stencil: inside every painted block, set the cells that match the
    # source's background cells back to background
    holes = [(sr, sc) for sr in range(h) for sc in range(w) if int(src[sr, sc]) == bgc]
    carve = [(r * h + sr, c * w + sc) for (r, c) in painted for (sr, sc) in holes]
    if carve:
        ops.append(bgc); sels.append(sel_of(carve))

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
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
                                f"for task b190f7f5"
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
                    f"Failed to build a complete episode for task b190f7f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b190f7f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
