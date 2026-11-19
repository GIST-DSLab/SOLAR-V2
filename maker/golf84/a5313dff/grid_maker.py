"""
ARC Task: a5313dff (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    def components(g, color):
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for i in range(h):
            for j in range(w):
                if g[i, j] == color and not seen[i, j]:
                    st = [(i, j)]
                    seen[i, j] = True
                    comp = []
                    border = False
                    while st:
                        r, c = st.pop()
                        comp.append((r, c))
                        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                            border = True
                        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == color and not seen[nr, nc]:
                                seen[nr, nc] = True
                                st.append((nr, nc))
                    comps.append((comp, border))
        return comps

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    inds = [(i, j) for i in range(-1, h + 1) for j in range(-1, w + 1)]

    def n_enclosed(g):
        return sum(1 for comp, b in components(g, bgc) if not b)

    while (tr < maxtr and succ < noccs) or n_enclosed(gi) == 0:
        tr += 1
        if tr > 2000:
            raise RuntimeError("generation failed")
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        li, lj = random.choice(inds)
        ins = [(li + a, lj + b) for a in range(1, oh - 1) for b in range(1, ow - 1)]
        if any(0 <= r < h and 0 <= c < w and gi[r, c] == fgc for r, c in ins):
            continue
        succ += 1
        for a in range(oh):
            for b in range(ow):
                if a in (0, oh - 1) or b in (0, ow - 1):
                    r, c = li + a, lj + b
                    if 0 <= r < h and 0 <= c < w:
                        gi[r, c] = fgc
        if random.choice((True, True, False)):
            k = random.randint(1, max(1, len(ins) // 2))
            for r, c in random.sample(ins, k):
                if 0 <= r < h and 0 <= c < w:
                    gi[r, c] = fgc
    go = gi.copy()
    for comp, b in components(gi, bgc):
        if not b:
            for r, c in comp:
                go[r, c] = 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background colour: read from the demonstrations (cells turned to 1 were background).
    bgc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            m = (ei != eo) & (eo == 1)
            cnt.update(ei[m].tolist())
        present = set(I.flatten().tolist())
        for col, _ in cnt.most_common():
            if col in present:
                bgc = col
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Find the background regions in I and check which ones touch the border.
    seen = np.zeros((hi, wi), dtype=bool)
    enclosed, outer = [], []
    for i in range(hi):
        for j in range(wi):
            if I[i, j] == bgc and not seen[i, j]:
                st = [(i, j)]
                seen[i, j] = True
                comp = []
                border = False
                while st:
                    r, c = st.pop()
                    comp.append((r, c))
                    if r == 0 or c == 0 or r == hi - 1 or c == wi - 1:
                        border = True
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < hi and 0 <= nc < wi and I[nr, nc] == bgc and not seen[nr, nc]:
                            seen[nr, nc] = True
                            st.append((nr, nc))
                (outer if border else enclosed).append(comp)

    ops, sels = [], []
    if enclosed and 1 + len(outer) < len(enclosed):
        # Larger stroke: paint every background cell 1 (selection = all cells of
        # colour bgc), then flood each border-touching region back to bgc.
        all_bg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
        ops.append(1); sels.append(sel_of(all_bg))
        for comp in outer:
            r0, c0 = comp[0]
            ops.append(10 + bgc); sels.append([r0, c0, 0, 0])
    else:
        # One FloodFill1 per enclosed background region.
        for comp in enclosed:
            r0, c0 = comp[0]
            ops.append(11); sels.append([r0, c0, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task a5313dff"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a5313dff"
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
                                f"for task a5313dff"
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
                    f"Failed to build a complete episode for task a5313dff "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a5313dff-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
