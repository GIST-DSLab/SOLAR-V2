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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc, **kw) -> dict:
    # original uses (10, 31) because one row/col is removed at the end
    h = _unifint(diff_lb, diff_ub, (10, max_h + 1))
    w = _unifint(diff_lb, diff_ub, (10, max_w + 1))
    oh = _unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = _unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = np.full((oh, ow), bgc, dtype=int)
    sg[oh - 1, ow - 1] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != (oh - 1, ow - 1)]
    hi_n = max(1, int((2 / 3) * oh * ow))
    while True:
        ncells = _unifint(diff_lb, diff_ub, (1, hi_n))
        cells = random.sample(reminds, ncells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if not (ncells == 4 and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1):
            break
    for r, c in cells:
        sg[r, c] = objc
    GG = np.full((2 * oh + 1, 2 * ow + 1), bgc, dtype=int)
    GG[:oh, :ow] = sg
    GG[:oh, ow + 1:] = np.fliplr(sg)
    GG[oh + 1:, :ow] = np.flipud(sg)
    GG[oh + 1:, ow + 1:] = np.flipud(np.fliplr(sg))
    GG[oh, ow] = cc
    canv = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = canv.copy()
    go[loci:loci + 2 * oh + 1, locj:locj + 2 * ow + 1] = GG
    gi = canv.copy()
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    pts = np.argwhere(gi == cc)
    ci = (pts[:, 0].min() + pts[:, 0].max()) // 2
    cj = (pts[:, 1].min() + pts[:, 1].max()) // 2
    gi = np.delete(np.delete(gi, ci, 0), cj, 1)
    go = np.delete(np.delete(go, ci, 0), cj, 1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # centre marker: the colour whose cells form exactly a 2x2 square
    ccol = None
    for col in set(I.flatten().tolist()) - {bgc}:
        p = np.argwhere(I == col)
        if len(p) == 4 and np.ptp(p[:, 0]) == 1 and np.ptp(p[:, 1]) == 1:
            ccol = col
            break
    cp = np.argwhere(I == ccol)
    r0, c0 = int(cp[:, 0].min()), int(cp[:, 1].min())
    objc = [c for c in set(I.flatten().tolist()) if c not in (bgc, ccol)][0]
    cells = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    mr = lambda r: 2 * r0 + 1 - r
    mc = lambda c: 2 * c0 + 1 - c
    ra, rb = min(r for r, _ in cells), max(r for r, _ in cells)
    ca, cb = min(c for _, c in cells), max(c for _, c in cells)
    bh, bw = rb - ra, cb - ca
    ops, sels = [], []
    # (mirror rows?, mirror cols?) for the three missing quadrants
    targets = [(True, False), (False, True), (True, True)]
    if bgc != 0 and ccol != 0 and objc != 0:
        # pattern bbox is a fully non-zero rectangle -> CopyI / Paste / Flip in place
        ops.append(28); sels.append([ra, ca, bh, bw])  # full rect: pattern bbox
        for fr, fc in targets:
            dr = mr(rb) if fr else ra
            dc = mc(cb) if fc else ca
            box = [dr, dc, bh, bw]  # full rectangle: mirrored pattern bbox
            ops.append(30); sels.append([dr, dc, 0, 0])
            if fc:
                ops.append(26); sels.append(box)
            if fr:
                ops.append(27); sels.append(box)
    else:
        # 0 is a semantic colour here (copy/paste would drop it): paint each reflected copy
        for fr, fc in targets:
            tgt = [(mr(r) if fr else r, mc(c) if fc else c) for r, c in cells]
            ops.append(objc); sels.append(sel_of(tgt))
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
