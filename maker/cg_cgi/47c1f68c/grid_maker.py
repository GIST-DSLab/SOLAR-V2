"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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

SCF_VARIANTS = ["identity", "hmirror", "vmirror", "both"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SCF_VARIANTS):
        examples = [{"scf": v} for v in SCF_VARIANTS]
        examples += [{"scf": random.choice(SCF_VARIANTS)}
                     for _ in range(n_ex - len(SCF_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"scf": v} for v in random.sample(SCF_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc, scf=None) -> dict:
    if scf is None:
        scf = random.choice(SCF_VARIANTS)

    def unif(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(14, (max_w - 1) // 2))
    h = unif(diff_lb, diff_ub, (2, hmax))
    w = unif(diff_lb, diff_ub, (2, wmax))
    nc = max(1, min(h * w - 1, unif(diff_lb, diff_ub, (1, h * w - 1))))

    allcells = [(r, c) for r in range(h) for c in range(w)]
    bx = set(allcells)
    start = random.choice(allcells)
    obj = {start}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        dns.add((r + dr, c + dc))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)

    objgrid = [[objc if (r, c) in obj else bgc for c in range(w)] for r in range(h)]
    blank = [[bgc] * w for _ in range(h)]
    top = [objgrid[r] + [linc] + blank[r] for r in range(h)]
    mid = [[linc] * (2 * w + 1)]
    bot = [blank[r] + [linc] + blank[r] for r in range(h)]
    gi = top + mid + bot

    P = [[linc if (r, c) in obj else bgc for c in range(w)] for r in range(h)]
    tophalf = [P[r] + P[r][::-1] for r in range(h)]
    go = tophalf + tophalf[::-1]

    if scf in ("hmirror", "both"):
        gi = gi[::-1]
        go = go[::-1]
    if scf in ("vmirror", "both"):
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # the divider cross sits in the middle row/col, splitting I into four h x w panels
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, w])

    bgc, src = None, None
    for (r0, c0) in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()          # an empty panel reveals the background
        else:
            src = (r0, c0)            # the one panel carrying the object
    r0, c0 = src
    panel = I[r0:r0 + h, c0:c0 + w]
    objc = [v for v in set(panel.flatten().tolist()) if v != bgc][0]

    ops, sels = [], []

    # 1. stash the object's panel straight from the input (whole rectangle)
    ops.append(28); sels.append([r0, c0, h - 1, w - 1])
    # 2. resize the canvas to the 2x2 mosaic of panels
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])

    # 3. mirror the raw panel into all four quadrants; the panel keeps the corner
    #    it occupied in I, the others are its reflections (whole-quadrant flips)
    fh0, fv0 = c0 > 0, r0 > 0
    cur = {}
    for qr, qc, fh, fv in [(0, 0, fh0, fv0), (0, w, not fh0, fv0),
                           (h, 0, fh0, not fv0), (h, w, not fh0, not fv0)]:
        q = panel.copy()
        if (qr, qc) != (0, 0) or src != (0, 0):
            ops.append(30); sels.append([qr, qc, 0, 0])
        if fh:
            nq = q[:, ::-1]
            if not np.array_equal(nq, q):
                ops.append(26); sels.append([qr, qc, h - 1, w - 1])
            q = nq
        if fv:
            nq = q[::-1, :]
            if not np.array_equal(nq, q):
                ops.append(27); sels.append([qr, qc, h - 1, w - 1])
            q = nq
        cur[(qr, qc)] = q

    # 4. recolour the mirrored object in the divider colour, one connected
    #    blob of the object colour on the current mosaic at a time
    G = np.zeros((2 * h, 2 * w), dtype=int)
    for (qr, qc), q in cur.items():
        G[qr:qr + h, qc:qc + w] = q
    seen = np.zeros(G.shape, dtype=bool)
    for r in range(2 * h):
        for c in range(2 * w):
            if G[r, c] != objc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < 2 * h and 0 <= nx < 2 * w and not seen[ny, nx]
                            and G[ny, nx] == objc):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ops.append(10 + linc); sels.append([r, c, 0, 0])

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
