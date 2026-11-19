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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of

try:
    from dsl import *
except Exception:
    pass


# ── sample_colors ─────────────────────────────────────────────────────────────
def sample_colors(num_examples=None) -> dict:
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


# ── generate ──────────────────────────────────────────────────────────────────
def generate(diff_lb, diff_ub, max_h, max_w,
             bgc=None, pc=None, c1=None, c2=None, c3=None, c4=None) -> dict:
    if None in (bgc, pc, c1, c2, c3, c4):
        bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    max_h = max(12, min(30, max_h))
    max_w = max(12, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (12, max_h))
    w = unifint(diff_lb, diff_ub, (12, max_w))
    oh = unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    ln3 = connect((0, 1), (0, ow - 2))
    ln4 = connect((oh - 1, 1), (oh - 1, ow - 2))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, c3, ln3)
    go = fill(go, c4, ln4)
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi,
                   lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    gi = paint(gi, plcdB)
    gi = fill(gi, pc, plcdA)
    objA = shift(objA, (1, 1))
    objs = objects(go, T, F, T)
    for ij in objA:
        manhs = {obj: manhattan(obj, {ij}) for obj in objs}
        manhsl = list(manhs.values())
        mmh = min(manhsl)
        if manhsl.count(mmh) == 1:
            col = color([o for o, mnh in manhs.items() if mmh == mnh][0])
        else:
            col = pc
        go = fill(go, col, {ij})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


# ── derive_operations ─────────────────────────────────────────────────────────
def _parse(I, bgc):
    H, W = I.shape
    cells = {}
    for r in range(H):
        for c in range(W):
            v = int(I[r, c])
            if v != bgc:
                cells.setdefault(v, []).append((r, c))
    lines, blob_cols = {}, []
    for col, cs in cells.items():
        rows = {r for r, _ in cs}
        cols = {c for _, c in cs}
        if len(rows) == 1 and len(cs) >= 2 and max(cols) - min(cols) + 1 == len(cs):
            lines[col] = ("h", cs)
        elif len(cols) == 1 and len(cs) >= 2 and max(rows) - min(rows) + 1 == len(cs):
            lines[col] = ("v", cs)
        else:
            blob_cols.append(col)
    if len(lines) != 4 or len(blob_cols) != 1:
        return None
    allc = [p for _, cs in lines.values() for p in cs]
    r0 = min(r for r, _ in allc); r1 = max(r for r, _ in allc)
    c0 = min(c for _, c in allc); c1 = max(c for _, c in allc)
    info = {"r0": r0, "c0": c0, "oh": r1 - r0 + 1, "ow": c1 - c0 + 1}
    for col, (kind, cs) in lines.items():
        if kind == "h":
            info["ctop" if cs[0][0] == r0 else "cbot"] = col
        else:
            info["cleft" if cs[0][1] == c0 else "cright"] = col
    if not all(k in info for k in ("ctop", "cbot", "cleft", "cright")):
        return None
    pc = blob_cols[0]
    blob = cells[pc]
    info["pc"] = pc
    info["blob"] = blob
    info["br"] = min(r for r, _ in blob)
    info["bc"] = min(c for _, c in blob)
    return info


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    info = None
    for bgc, _ in Counter(I.flatten().tolist()).most_common():
        info = _parse(I, bgc)
        if info is not None:
            break

    r0, c0 = info["r0"], info["c0"]
    oh, ow = info["oh"], info["ow"]
    pc = info["pc"]
    blob = list(info["blob"])

    # The blob is moved as a single unit so that its top-left lands on the box's interior (1,1).
    dr = (r0 + 1) - info["br"]
    dc = (c0 + 1) - info["bc"]

    ops, sels = [], []
    dest = [(r + dr, c + dc) for r, c in blob]

    if pc != 0:
        # Grab the blob once, then keep sliding it with empty selections.
        # The blob always starts outside the box, so the cells it leaves behind
        # get cropped away later and need no repair.
        steps = []
        steps += [20 if dr < 0 else 21] * abs(dr)
        steps += [23 if dc < 0 else 22] * abs(dc)
        for k, op in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(blob) if k == 0 else sel_of([]))
    else:
        # A blob of colour 0 can't be grabbed (Move/Copy skip 0), so stamp it into place instead.
        ops.append(0)
        sels.append(sel_of(dest))

    # The box's diagonals split its interior into four triangles, one per wall.
    # Blob cells inside a triangle take that wall's colour.
    # Cells on a diagonal are equally near two walls and keep the blob colour.
    walls = [("top", info["ctop"]), ("bottom", info["cbot"]),
             ("left", info["cleft"]), ("right", info["cright"])]
    claim = {}
    for (gr, gc) in dest:
        lr, lc = gr - r0, gc - c0
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        claim[(gr, gc)] = win[0] if len(win) == 1 else None
    for name, col in walls:
        grp = sorted(p for p in dest if claim[p] == name)
        if grp:
            ops.append(int(col))
            sels.append(sel_of(grp))

    # Keep only the box. This selection is exactly that full rectangle, background included.
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    ops.append(34)
    sels.append([0, 0, oh - 1, ow - 1])
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
