"""
ARC Task: 4938f0c2 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


# which quadrant the single input pattern sits in is set by the generator's final
# rotation: 0 = identity, 1 = rot90 CW, 2 = rot180, 3 = rot90 CCW
_ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    cc = random.choice([c for c in cols if c != bgc])
    objc = random.choice([c for c in cols if c not in (bgc, cc)])
    n_ex = num_examples if num_examples else 3
    variants = [{"rot": r} for r in _ROTS]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cc": cc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(_ROTS)
    transposed = rot in (1, 3)
    # the final grids drop one row and one column; a 90 degree turn swaps the axes
    lim_h = (max_w if transposed else max_h) + 1
    lim_w = (max_h if transposed else max_w) + 1
    h = _unifint(diff_lb, diff_ub, (10, max(10, min(31, lim_h))))
    w = _unifint(diff_lb, diff_ub, (10, max(10, min(31, lim_w))))
    oh = _unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = _unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))

    sg = np.full((oh, ow), bgc, dtype=int)
    locc = (oh - 1, ow - 1)
    sg[locc] = cc
    reminds = [(r, c) for r in range(oh) for c in range(ow) if (r, c) != locc]
    hi_n = max(1, int((2 / 3) * oh * ow))
    while True:
        ncells = _unifint(diff_lb, diff_ub, (1, min(hi_n, len(reminds))))
        cells = random.sample(reminds, ncells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        shp = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        if not (ncells == 4 and shp == (2, 2)):
            break
    for r, c in cells:
        sg[r, c] = objc

    GG = np.full((2 * oh + 1, 2 * ow + 1), bgc, dtype=int)
    GG[:oh, :ow] = sg
    GG[:oh, ow + 1:] = sg[:, ::-1]
    GG[oh + 1:, :ow] = sg[::-1, :]
    GG[oh + 1:, ow + 1:] = sg[::-1, ::-1]
    GG[oh, ow] = cc

    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + 2 * oh + 1, locj:locj + 2 * ow + 1] = GG
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc

    if rot == 1:
        gi, go = np.rot90(gi, 3), np.rot90(go, 3)
    elif rot == 2:
        gi, go = np.rot90(gi, 2), np.rot90(go, 2)
    elif rot == 3:
        gi, go = np.rot90(gi, 1), np.rot90(go, 1)

    # delete the centre row/column of the marker cross -> marker becomes a 2x2 block
    rs, cs = np.where(gi == cc)
    ccpi = int(rs.min()) + (int(rs.max()) - int(rs.min()) + 1) // 2
    ccpj = int(cs.min()) + (int(cs.max()) - int(cs.min()) + 1) // 2
    keep_r = [r for r in range(gi.shape[0]) if r != ccpi]
    keep_c = [c for c in range(gi.shape[1]) if c != ccpj]
    gi = gi[np.ix_(keep_r, keep_c)]
    go = go[np.ix_(keep_r, keep_c)]

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit = [0, 0, O.shape[0] - 1, O.shape[1] - 1]

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # the mirror centre: the only colour occupying exactly a solid 2x2 block
    # (the generator explicitly forbids the pattern colour from doing that)
    cc = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        rs, cs = np.where(I == col)
        if int(rs.max() - rs.min()) == 1 and int(cs.max() - cs.min()) == 1:
            cc, R, C = int(col), int(rs.min()), int(cs.min())
            break
    rest = [c for c in cnt if c != bgc and c != cc]
    if cc is None or not rest:
        return [34], [submit]
    objc = int(rest[0])

    cells = np.argwhere(I == objc)
    r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
    c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())

    def mr(r):
        return 2 * R + 1 - r          # mirror across the block's horizontal axis

    def mc(c):
        return 2 * C + 1 - c          # mirror across the block's vertical axis

    if objc == 0:
        # ARCLE's clipboard treats 0 as "nothing", so Copy/Paste cannot carry this
        # pattern at all: draw its mirror images instead.
        h_img = [(r, mc(c)) for r, c in cells]
        ops.append(0)
        sels.append(sel_of(h_img))
        v_img = [(mr(r), c) for r, c in cells] + [(mr(r), mc(c)) for r, c in cells]
        ops.append(0)
        sels.append(sel_of(v_img))
        ops.append(34)
        sels.append(submit)
        return ops, sels

    G = I.copy()

    def stamp(src, dr, dc, flip):
        """Paste the clipboard block at (dr, dc), then mirror it in place there."""
        sh, sw = src.shape
        ops.append(30)
        sels.append([dr, dc, 0, 0])
        dest = G[dr:dr + sh, dc:dc + sw]
        np.copyto(dest, src, where=(src != 0))
        # Paste is transparent to 0s: wherever the clipboard could not carry a 0
        # the destination still shows stale content (e.g. a marker cell) that the
        # mirror would drag to the wrong place. Fill those cells in.
        gap = [(dr + i, dc + j) for i in range(sh) for j in range(sw)
               if dest[i, j] != src[i, j]]
        if gap:
            ops.append(0)
            sels.append(sel_of(gap))
            for r, c in gap:
                G[r, c] = 0
        mirrored = np.fliplr(dest) if flip == 26 else np.flipud(dest)
        if not np.array_equal(mirrored, dest):
            # whole-rectangle mirror: the intended cells ARE this full block
            ops.append(flip)
            sels.append([dr, dc, sh - 1, sw - 1])
            np.copyto(dest, mirrored)

    # 1. mirror the pattern across the marker's vertical axis
    src = I[r0:r1 + 1, c0:c1 + 1].copy()
    ops.append(28)                                   # CopyI: the pattern's own block
    sels.append([r0, c0, r1 - r0, c1 - c0])
    stamp(src, r0, mc(c1), 26)

    # 2. mirror the resulting pair across the marker's horizontal axis
    cl, cr = min(c0, mc(c1)), max(c1, mc(c0))
    src2 = G[r0:r1 + 1, cl:cr + 1].copy()
    ops.append(29)                                   # CopyO: both halves as they stand
    sels.append([r0, cl, r1 - r0, cr - cl])
    stamp(src2, mr(r1), cl, 27)

    ops.append(34)
    sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
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
                                f"for task 4938f0c2"
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
                    f"Failed to build a complete episode for task 4938f0c2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4938f0c2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
