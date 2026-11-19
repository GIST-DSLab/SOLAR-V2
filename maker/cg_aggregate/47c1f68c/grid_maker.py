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
from maker.sel_helpers import sel_of

CORNERS = ["tl", "tr", "bl", "br"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(CORNERS):
        ex = list(CORNERS) + [random.choice(CORNERS) for _ in range(n_ex - len(CORNERS))]
        random.shuffle(ex)
    else:
        ex = random.sample(CORNERS, n_ex)
    plan = [{"corner": c} for c in ex] + [{"corner": random.choice(ex)}]
    return {"bgc": bgc, "linc": linc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, objc=None,
             corner=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    cols = list(range(1, 10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    if objc is None or objc in (bgc, linc):
        objc = random.choice([c for c in cols if c not in (bgc, linc)])
    if corner is None:
        corner = random.choice(CORNERS)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_h - 1) // 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_w - 1) // 2))))
    nc = unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)
    panel = np.full((h, w), bgc, dtype=int)
    for (i, j) in obj:
        panel[i, j] = objc
    canv = np.full((h, w), bgc, dtype=int)
    vl = np.full((h, 1), linc, dtype=int)
    gi1 = np.hstack([panel, vl, canv])
    gi2 = np.hstack([canv, vl, canv])
    gi = np.vstack([gi1, np.full((1, 2 * w + 1), linc, dtype=int), gi2])
    go = np.hstack([panel, np.fliplr(panel)])
    go = np.vstack([go, np.flipud(go)])
    go[go == objc] = linc
    if corner in ("tr", "br"):
        gi = np.fliplr(gi); go = np.fliplr(go)
    if corner in ("bl", "br"):
        gi = np.flipud(gi); go = np.flipud(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): the object panel, recoloured as ONE unit
    # (all its non-background cells take the divider colour), is mirrored sideways into a
    # half-strip, and that whole strip is then mirrored as ONE unit to the other half.
    # Different from the maker, which floodfills blob by blob and pastes four quadrants separately.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, 0])                       # colour of the dividing cross
    bgc, src = None, None
    for (r0, c0) in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()                  # an empty panel shows the background
        else:
            src = (r0, c0)                    # the panel that holds the object
    r0, c0 = src
    ops, sels = [], []

    # 1. Output frame = input without the cross: crop the 2h x 2w window (a full
    #    rectangle) that keeps the object panel in its own corner.
    rs, cs = (0 if r0 == 0 else 1), (0 if c0 == 0 else 1)
    G = I[rs:rs + 2 * h, cs:cs + 2 * w].copy()
    ops.append(33); sels.append([rs, cs, 2 * h - 1, 2 * w - 1])
    qr, qc = (0 if r0 == 0 else h), (0 if c0 == 0 else w)

    # 2. Recolour the object (every non-background cell of the panel) to the divider colour.
    panel = G[qr:qr + h, qc:qc + w]
    cells = [(qr + r, qc + c) for r in range(h) for c in range(w) if panel[r, c] != bgc]
    ops.append(linc); sels.append(sel_of(cells))
    for (r, c) in cells:
        G[r, c] = linc

    def emit_mirror(src_rect, dst_rc, flip_op):
        sr, sc, sh, sw = src_rect
        clip = G[sr:sr + sh, sc:sc + sw].copy()
        dr, dc = dst_rc
        ops.append(29); sels.append([sr, sc, sh - 1, sw - 1])           # copy the current unit (full rect)
        if not np.array_equal(G[dr:dr + sh, dc:dc + sw], clip):
            ops.append(30); sels.append([dr, dc, 0, 0])                 # paste it beside itself
            G[dr:dr + sh, dc:dc + sw] = clip
        fl = np.fliplr(clip) if flip_op == 26 else np.flipud(clip)
        if not np.array_equal(fl, clip):
            ops.append(flip_op); sels.append([dr, dc, sh - 1, sw - 1])  # mirror the pasted block in place
            G[dr:dr + sh, dc:dc + sw] = fl

    # 3. Mirror the panel across the vertical axis to make a symmetric half-strip.
    emit_mirror((qr, qc, h, w), (qr, w - qc), 26)
    # 4. Mirror that whole half-strip across the horizontal axis as one unit.
    emit_mirror((qr, 0, h, 2 * w), (h - qr, 0), 27)

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
