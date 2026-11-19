"""
ARC Task: f9012d9b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Tile colours are arbitrary and the rule (complete the periodic tiling inside
    # the 0-hole, output the hole) never depends on them; 0 is the hole marker.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kw) -> dict:
    def U(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return lo + int(round(d * (hi - lo)))
    cols = list(range(1, 10))
    hp = U(2, min(10, max_h // 3))
    wp = U(2, min(10, max_w // 3))
    nc = U(1, 9)
    ccols = random.sample(cols, nc)
    tile = np.array([[random.choice(ccols) for _ in range(wp)] for _ in range(hp)])
    numhp = U(3, max_h // hp)
    numwp = U(3, max_w // wp)
    gi = np.tile(tile, (numhp, numwp))
    gi = gi[:gi.shape[0] - random.randint(0, hp), :]
    gi = gi[:, :gi.shape[1] - random.randint(0, wp)]
    h, w = gi.shape
    sgh = U(1, h - hp - 1)
    sgw = U(1, w - wp - 1)
    loci = random.randint(0, h - sgh)
    locj = random.randint(0, w - sgw)
    go = gi[loci:loci + sgh, locj:locj + sgw].copy()
    gi = gi.copy()
    gi[loci:loci + sgh, locj:locj + sgw] = 0
    # every hole cell's tile residue must still be visible somewhere (else unsolvable)
    seen = {(r % hp, c % wp) for r in range(h) for c in range(w) if gi[r, c] != 0}
    if any((r % hp, c % wp) not in seen
           for r in range(loci, loci + sgh) for c in range(locj, locj + sgw)):
        return generate(diff_lb, diff_ub, max_h, max_w, **kw)
    mfs = [lambda g: g, lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2),
           lambda g: np.rot90(g, 1), lambda g: g.T, lambda g: g[:, ::-1],
           lambda g: g[::-1, :], lambda g: np.rot90(g, 2).T]
    mf = random.choice(mfs)
    gi2, go2 = mf(gi), mf(go)
    if gi2.shape[0] > max_h or gi2.shape[1] > max_w:
        return generate(diff_lb, diff_ub, max_h, max_w, **kw)
    return {"input": gi2.tolist(), "output": go2.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    holes = [(r, c) for r in range(h) for c in range(w) if I[r, c] == 0]
    r0 = min(r for r, _ in holes); r1 = max(r for r, _ in holes)
    c0 = min(c for _, c in holes); c1 = max(c for _, c in holes)
    sh, sw = r1 - r0 + 1, c1 - c0 + 1

    # smallest row/col period consistent with every visible (non-zero) cell pair
    def period(axis):
        n = h if axis == 0 else w
        for p in range(1, n):
            A = I[:-p, :] if axis == 0 else I[:, :-p]
            B = I[p:, :] if axis == 0 else I[:, p:]
            m = (A != 0) & (B != 0)
            if np.all(A[m] == B[m]):
                return p
        return n
    ph, pw = period(0), period(1)

    G = I.copy()
    ops, sels = [], []
    shifts = [(k * ph, m * pw)
              for k in range(-h // ph - 1, h // ph + 2)
              for m in range(-w // pw - 1, w // pw + 2) if (k, m) != (0, 0)]
    shifts.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s))
    for dr, dc in shifts:
        if not np.any(G[r0:r1 + 1, c0:c1 + 1] == 0):
            break
        # period-shifted copy of the hole rectangle, clipped to the grid
        sr0, sr1 = max(0, r0 + dr), min(h - 1, r1 + dr)
        sc0, sc1 = max(0, c0 + dc), min(w - 1, c1 + dc)
        if sr0 > sr1 or sc0 > sc1:
            continue
        src = I[sr0:sr1 + 1, sc0:sc1 + 1]
        dr0, dc0 = sr0 - dr, sc0 - dc
        dst = G[dr0:dr0 + src.shape[0], dc0:dc0 + src.shape[1]]
        if not np.any((dst == 0) & (src != 0)):
            continue  # this paste would fill nothing new
        # bbox OK: the intended copy region is exactly this full rectangle of the input
        ops.append(28); sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
        ops.append(30); sels.append([dr0, dc0, 0, 0])
        dst[src != 0] = src[src != 0]
    # bbox OK: crop canvas to exactly the (now filled) hole rectangle
    ops.append(33); sels.append([r0, c0, sh - 1, sw - 1])
    ops.append(34); sels.append([0, 0, sh - 1, sw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
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
                                f"for task f9012d9b"
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
                    f"Failed to build a complete episode for task f9012d9b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f9012d9b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
