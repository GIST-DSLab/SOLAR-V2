"""
ARC Task: 9aec4887 (RE-ARC) — LLM-generated grid_maker
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
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((oh, ow), bgc, dtype=int)
    go[1:oh - 1, 0] = c1
    go[1:oh - 1, ow - 1] = c2
    go[0, 1:ow - 1] = c3
    go[oh - 1, 1:ow - 1] = c4
    ih, iw = oh - 2, ow - 2
    bounds = {(i, j) for i in range(ih) for j in range(iw)}
    objA = {random.choice(sorted(bounds))}

    def nbrs(S):
        out = set()
        for (i, j) in S:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    ncells = _unifint(diff_lb, diff_ub, (1, (ih * iw) // 2))
    for _ in range(ncells - 1):
        cand = sorted((bounds - objA) & nbrs(objA))
        if not cand:
            break
        objA.add(random.choice(cand))

    def shp(S):
        rs = [i for i, _ in S]; cs = [j for _, j in S]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(objA) != (ih, iw):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    boxcells = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in boxcells and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    gi[loci:loci + oh, locj:locj + ow] = go
    for (i, j) in objA:
        gi[li + i, lj + j] = pc
    for (i, j) in objA:
        lr, lc = i + 1, j + 1
        d = {c3: lr, c4: oh - 1 - lr, c1: lc, c2: ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        go[lr, lc] = win[0] if len(win) == 1 else pc
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, color):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if I[r, c] == color and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; comp = []
                while st:
                    a, b = st.pop(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and not seen[x, y] and I[x, y] == color:
                            seen[x, y] = True; st.append((x, y))
                comps.append(comp)
    return comps


def _parse(I, bgc):
    H, W = I.shape
    best = None
    # box interior = largest full-rectangle background component strictly inside the grid
    for comp in _components(I, bgc):
        rs = [r for r, _ in comp]; cs = [c for _, c in comp]
        r1, c1 = min(rs), min(cs)
        h = max(rs) - r1 + 1; w = max(cs) - c1 + 1
        if len(comp) != h * w:
            continue
        if r1 < 1 or c1 < 1 or r1 + h >= H or c1 + w >= W:
            continue
        if best is None or len(comp) > best[4]:
            best = (r1, c1, h, w, len(comp))
    if best is None:
        return None
    r1, c1, ih, iw, _ = best
    r0, c0, oh, ow = r1 - 1, c1 - 1, ih + 2, iw + 2
    ctop, cbot = I[r0, c1], I[r0 + oh - 1, c1]
    cleft, cright = I[r1, c0], I[r1, c0 + ow - 1]
    blob = [(r, c) for r in range(H) for c in range(W)
            if I[r, c] != bgc and not (r0 <= r < r0 + oh and c0 <= c < c0 + ow)]
    if not blob:
        return None
    pc = I[blob[0]]
    br = min(r for r, _ in blob); bc = min(c for _, c in blob)
    return dict(r0=r0, c0=c0, oh=oh, ow=ow, pc=pc, blob=blob, br=br, bc=bc,
                ctop=ctop, cbot=cbot, cleft=cleft, cright=cright)


def derive_operations(I, O=None):
    I = np.asarray(I, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    info = _parse(I, bgc)
    r0, c0, oh, ow = info["r0"], info["c0"], info["oh"], info["ow"]

    # blob carried into the box interior: its bbox top-left lands on (1,1) of the box
    claim = {}
    for (r, c) in info["blob"]:
        lr, lc = r - info["br"] + 1, c - info["bc"] + 1
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        claim[(r0 + lr, c0 + lc)] = win[0] if len(win) == 1 else None

    ops, sels = [], []
    # each wall colours the blob cells strictly nearest to it (one stroke per wall)
    for name, col in (("top", info["ctop"]), ("bottom", info["cbot"]),
                      ("left", info["cleft"]), ("right", info["cright"])):
        grp = sorted(p for p, k in claim.items() if k == name)
        if grp:
            ops.append(int(col)); sels.append(sel_of(grp))
    # cells equidistant from two walls keep the blob's own colour
    ties = sorted(p for p, k in claim.items() if k is None)
    if ties:
        ops.append(int(info["pc"])); sels.append(sel_of(ties))
    # keep only the box (selection is exactly that full rectangle, background included)
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
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
                                f"for task 9aec4887"
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
                    f"Failed to build a complete episode for task 9aec4887 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9aec4887-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
