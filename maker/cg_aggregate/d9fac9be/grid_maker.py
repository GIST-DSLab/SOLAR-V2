"""
ARC Task: d9fac9be (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, ringc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "ringc": ringc}


def _ring_centers(g, bgc):
    h, w = g.shape
    found = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            ctr = int(g[r, c])
            if ctr == bgc:
                continue
            win = g[r - 1:r + 2, c - 1:c + 2].flatten().tolist()
            ring = set(win[:4] + win[5:])
            if len(ring) != 1:
                continue
            rc = ring.pop()
            if rc == bgc or rc == ctr:
                continue
            found.append((r, c))
    return found


def _ring_occ(g, col):
    h, w = g.shape
    occ = []
    for i in range(h - 2):
        for j in range(w - 2):
            win = g[i:i + 3, j:j + 3]
            ok = True
            for dr in range(3):
                for dc in range(3):
                    if (dr, dc) != (1, 1) and win[dr, dc] != col:
                        ok = False
            if ok:
                occ.append((i, j))
    return occ


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, ringc=None, **kw) -> dict:
    if bgc is None or noisec is None or ringc is None:
        bgc, noisec, ringc = random.sample(range(10), 3)

    def unifint(lo, hi):
        lo_v = lo + (hi - lo) * diff_lb
        hi_v = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, int(round(random.uniform(lo_v, hi_v)))))

    rng = [(dr, dc) for dr in range(3) for dc in range(3) if (dr, dc) != (1, 1)]
    for _ in range(200):
        h = unifint(6, max(6, max_h))
        w = unifint(6, max(6, max_w))
        h, w = min(h, max_h), min(w, max_w)
        gi = np.full((h, w), bgc, dtype=int)
        nn1 = unifint(1, max(1, (h * w) // 3 - 1))
        nn2 = unifint(1, max(1, (h * w) // 3 - 9))
        inds = [(r, c) for r in range(h) for c in range(w)]
        n1 = random.sample(inds, nn1)
        s1 = set(n1)
        rest = [x for x in inds if x not in s1]
        n2 = random.sample(rest, min(nn2, len(rest)))
        for r, c in n1:
            gi[r, c] = noisec
        for r, c in n2:
            gi[r, c] = ringc
        occ1 = _ring_occ(gi, noisec)
        occ2 = _ring_occ(gi, ringc)
        for (i, j) in occ1:
            dr, dc = random.choice(rng)
            gi[i + dr, j + dc] = random.choice((bgc, ringc))
        for (i, j) in occ2:
            dr, dc = random.choice(rng)
            gi[i + dr, j + dc] = random.choice((bgc, noisec))
        li = random.randint(0, h - 3)
        lj = random.randint(0, w - 3)
        for dr, dc in rng:
            gi[li + dr, lj + dc] = ringc
        gi[li + 1, lj + 1] = noisec
        if Counter(gi.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        if _ring_centers(gi, bgc) != [(li + 1, lj + 1)]:
            continue
        return {"input": gi.tolist(), "output": [[noisec]]}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    # Rule (read from I alone): exactly one cell is a non-background center whose
    # 8 neighbours share one other non-background colour. The answer is that
    # center pixel, so crop the canvas straight down to that one cell.
    # (The code-golf entry treats the answer as "the least frequent colour". The
    # generator does not guarantee that, so the ring-center reading is kept.
    # Either way the visible change is the same: one crop to one pixel.)
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cr, cc = 0, 0
    found = False
    for r in range(1, hi - 1):
        for c in range(1, wi - 1):
            ctr = int(I[r, c])
            if ctr == bgc:
                continue
            win = I[r - 1:r + 2, c - 1:c + 2].flatten().tolist()
            ring = set(win[:4] + win[5:])
            if len(ring) != 1:
                continue
            rc = ring.pop()
            if rc == bgc or rc == ctr:
                continue
            cr, cc, found = r, c, True
            break
        if found:
            break

    ops, sels = [], []
    # CropGrid onto the single ring-center cell (a full 1x1 rectangle).
    ops.append(33)
    sels.append([cr, cc, 0, 0])
    ops.append(34)
    sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
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
                                f"for task d9fac9be"
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
                    f"Failed to build a complete episode for task d9fac9be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d9fac9be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
