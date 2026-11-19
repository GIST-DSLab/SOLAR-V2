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
    cc = random.choice([c for c in cols if c != bgc])              # 2x2 marker block colour
    objc = random.choice([c for c in cols if c not in (bgc, cc)])  # picture colour
    n_ex = num_examples if num_examples else 3
    rots = [0, 1, 2, 3]                                            # which quadrant holds the picture
    if n_ex >= len(rots):
        examples = [{"rot": r} for r in rots]
        examples += [{"rot": random.choice(rots)} for _ in range(n_ex - len(rots))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(rots, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cc": cc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc, rot=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        b = max(a, b)
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(lo, min(hi, b))
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    rot = int(rot) % 4

    if rot % 2 == 1:
        hcap = max(10, min(31, int(max_w) + 1))
        wcap = max(10, min(31, int(max_h) + 1))
    else:
        hcap = max(10, min(31, int(max_h) + 1))
        wcap = max(10, min(31, int(max_w) + 1))

    h = unifint(diff_lb, diff_ub, (10, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
    oh = unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))

    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != (oh - 1, ow - 1)]
    maxn = max(1, int((2.0 / 3.0) * oh * ow))
    while True:
        ncells = unifint(diff_lb, diff_ub, (1, maxn))
        ncells = min(ncells, len(reminds))
        cells = random.sample(reminds, ncells)
        if len(cells) == 4:
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            if (max(rs) - min(rs)) == (max(cs) - min(cs)):
                continue
        break

    sg = [[bgc] * ow for _ in range(oh)]
    for (i, j) in cells:
        sg[i][j] = objc
    sg[oh - 1][ow - 1] = cc

    G1 = [list(r) for r in sg]
    G2 = [list(r[::-1]) for r in sg]
    G3 = [list(r) for r in sg[::-1]]
    G4 = [list(r[::-1]) for r in sg[::-1]]

    GG = []
    for i in range(oh):
        GG.append(G1[i] + [bgc] + G2[i])
    GG.append([bgc] * ow + [cc] + [bgc] * ow)
    for i in range(oh):
        GG.append(G3[i] + [bgc] + G4[i])

    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)

    go = [[bgc] * w for _ in range(h)]
    for i in range(2 * oh + 1):
        for j in range(2 * ow + 1):
            go[loci + i][locj + j] = GG[i][j]

    gi = [[bgc] * w for _ in range(h)]
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = sg[i][j]
    for i in range(h):
        for j in range(w):
            if go[i][j] == cc:
                gi[i][j] = cc

    ai = np.array(gi, dtype=int)
    ao = np.array(go, dtype=int)
    if rot:
        ai = np.rot90(ai, rot)
        ao = np.rot90(ao, rot)

    pos = np.argwhere(ai == cc)
    r0, r1 = int(pos[:, 0].min()), int(pos[:, 0].max())
    c0, c1 = int(pos[:, 1].min()), int(pos[:, 1].max())
    ci = r0 + (r1 - r0 + 1) // 2
    cj = c0 + (c1 - c0 + 1) // 2
    ai = np.delete(np.delete(ai, ci, axis=0), cj, axis=1)
    ao = np.delete(np.delete(ao, ci, axis=0), cj, axis=1)

    return {"input": ai.tolist(), "output": ao.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def bg_of(g):
        g = np.asarray(g, dtype=int)
        return int(Counter(g.flatten().tolist()).most_common(1)[0][0])

    def markers(g):
        """colours whose whole cell set is exactly a solid 2x2 block -> {colour: (top,left)}"""
        g = np.asarray(g, dtype=int)
        b = bg_of(g)
        bycol = {}
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v = int(g[r, c])
                if v != b:
                    bycol.setdefault(v, []).append((r, c))
        out = {}
        for col, cs in bycol.items():
            if len(cs) == 4:
                rr = sorted({r for r, _ in cs})
                cc_ = sorted({c for _, c in cs})
                if len(rr) == 2 and len(cc_) == 2 and rr[1] == rr[0] + 1 and cc_[1] == cc_[0] + 1:
                    out[col] = (rr[0], cc_[0])
        return out

    full = [0, 0, hi - 1, wi - 1]
    bgc = bg_of(I)
    cand = markers(I)
    if not cand:
        return [34], [full]
    if len(cand) == 1:
        mcol = next(iter(cand))
    else:
        # marker role is fixed per episode: let the demonstrations settle it
        votes = Counter()
        for ex in (examples or []):
            ei = ex.get("input") if isinstance(ex, dict) else ex[0]
            if ei is None:
                continue
            for k in markers(ei):
                votes[k] += 1
        pool = [c for c in cand if votes.get(c, 0) > 0]
        mcol = max(pool, key=lambda c: votes[c]) if pool else sorted(cand)[0]

    br, bc = cand[mcol]
    ar2, ac2 = 2 * br + 1, 2 * bc + 1      # doubled mirror axes through the marker centre

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and int(I[r, c]) != mcol]
    if not obj:
        return [34], [full]
    objc = int(I[obj[0][0], obj[0][1]])
    r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
    c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)

    # row band of the picture, made symmetric about the marker's vertical axis
    bc0, bc1 = min(c0, ac2 - c1), max(c1, ac2 - c0)
    # rows made symmetric about the marker's horizontal axis
    br0, br1 = min(r0, ar2 - r1), max(r1, ar2 - r0)

    ops, sels = [], []
    # 1. reflect the picture across the vertical axis: FlipH the whole symmetric band
    #    (full rectangle intended: picture side + empty mirror side + symmetric marker cells)
    ops.append(26); sels.append([r0, bc0, r1 - r0, bc1 - bc0])
    # 2. the flip carried the original away -> stamp it back from the input
    if objc != 0:
        ops.append(28); sels.append([r0, c0, r1 - r0, c1 - c0])   # picture bbox in I (rectangle)
        ops.append(30); sels.append([r0, c0, 0, 0])
    else:
        ops.append(0); sels.append(sel_of(obj))                  # colour 0 cannot be pasted
    # 3. the band now holds both left/right copies: reflect it across the horizontal
    #    axis by flipping the whole symmetric block, then put the band back
    if objc != 0:
        ops.append(29); sels.append([r0, bc0, r1 - r0, bc1 - bc0])     # current band (rectangle)
        ops.append(27); sels.append([br0, bc0, br1 - br0, bc1 - bc0])  # full symmetric block
        ops.append(30); sels.append([r0, bc0, 0, 0])
    else:
        ops.append(27); sels.append([br0, bc0, br1 - br0, bc1 - bc0])
        band = sorted(set(obj) | {(r, ac2 - c) for r, c in obj})
        ops.append(0); sels.append(sel_of(band))

    ops.append(34); sels.append(full)
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
