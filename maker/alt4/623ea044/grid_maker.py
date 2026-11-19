"""
ARC Task: 623ea044 (RE-ARC) — LLM-generated grid_maker
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
    # The rule is "every dot shoots four diagonal rays"; only two colour roles exist.
    # Background is pinned to 0 so the dot field itself is the only non-transparent
    # content in the grid -- that is what lets the rays be built by re-stamping the
    # dot field (CopyI/Paste treat 0 as "nothing here").
    bgc = 0
    fgc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, fgc=None) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_w))))
    if fgc is None:
        fgc = choice(remove(bgc, colopts))
    g = canvas(bgc, (h, w))
    inds = totuple(asindices(g))
    card_bounds = (1, max(int(h * w * 0.1), 1))
    numdots = unifint(diff_lb, diff_ub, card_bounds)
    numdots = max(1, min(numdots, len(inds)))
    dots = sample(inds, numdots)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, fgc, dots)
    go = fill(gi, fgc, mapply(rbind(shoot, UP_RIGHT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, DOWN_LEFT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, UNITY), dots))
    go = fill(go, fgc, mapply(rbind(shoot, NEG_UNITY), dots))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Each dot emits four diagonal rays.

    Account of the construction: the whole ray field is the dot field itself,
    re-stamped over and over, one diagonal step further each time.  A ray family
    (say "down-right") is exactly  union_k { dots shifted by k*(1,1) }.  So for
    each of the four diagonal directions we take the input's dot field (CopyI --
    the input is pristine dots on a 0 background, so the copy is transparent
    everywhere except the dots) and Paste it one step further along that
    diagonal, repeatedly, until the field has slid off the grid.  No cell is
    painted individually; the ops are "the dots again, one step out".
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]

    ops, sels = [], []
    cur = I.copy()

    if dots and bgc == 0:
        # Four ray families, grown one diagonal step at a time.
        for dr, dc in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            for k in range(1, min(H, W)):
                # Source window of the INPUT that stays on the grid after the
                # k-step diagonal shift, and the corner it lands on.
                sr = max(0, -dr * k)
                sc = max(0, -dc * k)
                sh = H - k
                sw = W - k
                dest_r = max(0, dr * k)
                dest_c = max(0, dc * k)

                src = I[sr:sr + sh, sc:sc + sw]
                nxt = cur.copy()
                region = nxt[dest_r:dest_r + sh, dest_c:dest_c + sw]
                mask = src != 0                      # only the dots are carried
                region[mask] = src[mask]
                if np.array_equal(nxt, cur):
                    continue                          # this step adds nothing: skip it

                # bbox = exactly this full rectangle of the input (dots + 0s)
                ops.append(28); sels.append([sr, sc, sh - 1, sw - 1])   # CopyI dot field
                ops.append(30); sels.append([dest_r, dest_c, 0, 0])     # Paste one step out
                cur = nxt

    elif dots:
        # Safety net for a non-zero background (Copy/Paste is not transparent then):
        # draw each dot's two diagonal lines as whole lines.
        fgc = int(I[dots[0][0], dots[0][1]])
        for (r, c) in dots:
            for dr, dc in ((1, 1), (1, -1)):
                cells = []
                for s in (1, -1):
                    rr, cc = r + s * dr, c + s * dc
                    while 0 <= rr < H and 0 <= cc < W:
                        if cur[rr, cc] != fgc:
                            cells.append((rr, cc))
                        rr += s * dr; cc += s * dc
                if not cells:
                    continue
                for (rr, cc) in cells:
                    cur[rr, cc] = fgc
                try:
                    from maker.sel_helpers import sel_of
                    sel = sel_of(cells)
                except Exception:
                    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
                    sel = [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]
                ops.append(fgc); sels.append(sel)

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
                        f"num_examples+1 ({num_examples + 1}) for task 623ea044"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 623ea044"
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
                                f"for task 623ea044"
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
                    f"Failed to build a complete episode for task 623ea044 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"623ea044-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
