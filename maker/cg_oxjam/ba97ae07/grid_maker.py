"""
ARC Task: ba97ae07 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    v = lo + int(round(d * (hi - lo)))
    return max(lo, min(hi, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, acol=1, bcol=2, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])
    cap_h = min(max_h, max_w) if mirror else max_h
    cap_w = min(max_h, max_w) if mirror else max_w
    cap_h = max(3, min(30, cap_h))
    cap_w = max(3, min(30, cap_w))
    h = _unifint(diff_lb, diff_ub, 3, cap_h)
    w = _unifint(diff_lb, diff_ub, 3, cap_w)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    lineh = _unifint(diff_lb, diff_ub, 1, max(1, h // 3))
    linew = _unifint(diff_lb, diff_ub, 1, max(1, w // 3))
    loci = random.randint(1, h - lineh - 1)
    locj = random.randint(1, w - linew - 1)
    # input: horizontal band (acol) drawn first, vertical band (bcol) on top
    gi[loci:loci + lineh, :] = acol
    gi[:, locj:locj + linew] = bcol
    # output: horizontal band drawn on top
    go[:, locj:locj + linew] = bcol
    go[loci:loci + lineh, :] = acol
    if mirror:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                comps.append(cells)
    return comps


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the canvas colour. The generator guarantees it is the strict majority,
    # because each band covers at most a third of the grid.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [c for c in sorted(set(I.flatten().tolist())) if c != bgc]

    # Find the two bands. The one drawn underneath is cut in two by the crossing band,
    # so it has two pieces. The band drawn on top has only one piece.
    under, over, best_n = None, None, -1
    for c in fg:
        n = len(_components(I == c))
        if n > best_n:
            under, best_n = c, n
    others = [c for c in fg if c != under]
    over = others[0] if others else None

    if under is not None and over is not None:
        # Each band runs across the whole grid one way. Its bounding box is its full extent.
        ur, uc = np.nonzero(I == under)
        orr, oc = np.nonzero(I == over)
        u_r0, u_r1, u_c0, u_c1 = ur.min(), ur.max(), uc.min(), uc.max()
        o_r0, o_r1, o_c0, o_c1 = orr.min(), orr.max(), oc.min(), oc.max()
        # The crossing is where the two bands' extents overlap. The over-band covers it
        # now. Paint the under-band's colour back onto that patch so it runs unbroken.
        r0, r1 = max(u_r0, o_r0), min(u_r1, o_r1)
        c0, c1 = max(u_c0, o_c0), min(u_c1, o_c1)
        if r0 <= r1 and c0 <= c1:
            crossing = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            ops.append(int(under))
            sels.append(sel_of(crossing))

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
                        f"num_examples+1 ({num_examples + 1}) for task ba97ae07"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ba97ae07"
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
                                f"for task ba97ae07"
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
                    f"Failed to build a complete episode for task ba97ae07 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ba97ae07-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
