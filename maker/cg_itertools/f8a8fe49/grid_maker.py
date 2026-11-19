"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, boxc=2,
             transpose=None) -> dict:
    def unifint(a, b):
        if b <= a:
            return a
        lo = a + (b - a) * diff_lb
        hi = a + (b - a) * diff_ub
        return int(min(b, max(a, round(random.uniform(lo, hi)))))

    if transpose is None:
        transpose = random.choice([True, False])
    hb = min(30, max_w if transpose else max_h)
    wb = min(30, max_h if transpose else max_w)
    if hb < 10 or wb < 4:
        raise ValueError("grid too small")
    h = unifint(10, hb)
    w = unifint(4, wb)
    fullh = unifint(10, h)
    fullw = unifint(3, w)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    canvi = [[bgc] * fullw for _ in range(fullh)]   # pattern outside (-> output)
    canvo = [[bgc] * fullw for _ in range(fullh)]   # pattern inside  (-> input)
    objh = (fullh // 2 - 3) // 2
    line = objh + 1
    br = [(line, j) for j in range(fullw)] + [(line + 1, 0), (line + 1, fullw - 1)]
    cands = [(i, j) for i in range(objh) for j in range(1, fullw - 1)]
    total = len(cands)
    for _ in range(2):
        for (i, j) in br:
            canvi[i][j] = boxc
            canvo[i][j] = boxc
        ncellsd = unifint(0, total // 2)
        ncells = random.choice((ncellsd, total - ncellsd))
        ncells = min(max(1, ncells), total)
        cells = set(random.sample(cands, ncells))
        cells.add(random.choice([c for c in cands if c[0] == objh - 1]))
        for (i, j) in cells:
            canvi[i][j] = objc
            canvo[2 * line - i][j] = objc          # pure reflection through the box line (matches verifier)
        canvi = canvi[::-1]
        canvo = canvo[::-1]
    for i in range(fullh):
        for j in range(fullw):
            gi[loci + i][locj + j] = canvi[i][j]
            go[loci + i][locj + j] = canvo[i][j]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": go, "output": gi}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the golf entry, which carves the task differently from a
    single band flip per box line): every grid line that CROSSES a box line is
    handled on its own -- the short stretch of that crossing line centred on the
    box cell is reversed, which throws the pattern cells lying just inside the
    box out through the box cell to their mirror position outside.  The window
    radius is held fixed per box line (depth of the deepest inside pattern cell
    along that box line); crossing lines that carry no pattern on that side are
    left alone (reversing them would change nothing).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    bgc = int(I[0, 0])          # the canvas corner is always background

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})

    def area(c):
        b = bbox(c)
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    boxc = max(others, key=area)              # the box bbox strictly encloses the pattern
    objc = [c for c in others if c != boxc][0]
    r0, r1, c0, c1 = bbox(boxc)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))
    pat = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]

    ops, sels = [], []
    if horizontal:
        # box lines are rows r0, r1; each interior column crosses both of them
        mid = (r0 + r1) / 2.0
        for line, inside in ((r0, lambda r: r < mid), (r1, lambda r: r > mid)):
            side = [(r, c) for r, c in pat if inside(r)]
            m = max(abs(r - line) for r, _ in side)
            for c in sorted({c for _, c in side}):
                # full column segment centred on the box cell (background included) -> FlipV
                ops.append(27); sels.append([line - m, c, 2 * m, 0])
    else:
        # box lines are columns c0, c1; each interior row crosses both of them
        mid = (c0 + c1) / 2.0
        for line, inside in ((c0, lambda c: c < mid), (c1, lambda c: c > mid)):
            side = [(r, c) for r, c in pat if inside(c)]
            m = max(abs(c - line) for _, c in side)
            for r in sorted({r for r, _ in side}):
                # full row segment centred on the box cell (background included) -> FlipH
                ops.append(26); sels.append([r, line - m, 0, 2 * m])

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
