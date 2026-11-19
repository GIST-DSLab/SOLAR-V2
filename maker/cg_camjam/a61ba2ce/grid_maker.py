"""
ARC Task: a61ba2ce (RE-ARC) — LLM-generated grid_maker
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
    bgc, c1, c2, c3, c4 = random.sample(list(range(10)), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, c3=None, c4=None) -> dict:
    if bgc is None or None in (c1, c2, c3, c4):
        bgc, c1, c2, c3, c4 = random.sample(list(range(10)), 5)
    hub = max(4, min(15, max_h // 2))
    wub = max(4, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (4, hub))
    w = unifint(diff_lb, diff_ub, (4, wub))
    lociL = randint(2, h - 2)
    lociR = randint(2, h - 2)
    locjT = randint(2, w - 2)
    locjB = randint(2, w - 2)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(diff_lb, diff_ub, (2 * h, max(2 * h, min(30, max_h))))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max(2 * w, min(30, max_w))))
    gi = canvas(bgc, (fullh, fullw))
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    for _ in range(100):
        inds = asindices(gi)
        locs = []
        for o, c in zip(objs, ocols):
            cands = sfilter(inds, lambda ij: shift(o, ij).issubset(inds))
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            locs.append(loc)
            inds = inds - shift(o, loc)
        if len(locs) == 4:
            break
    else:
        raise ValueError("placement failed")
    for o, c, l in zip(objs, ocols, locs):
        gi = fill(gi, c, shift(o, l))
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf entry): each L piece's BOUNDING BOX is one tile,
    # and the output is those four tiles placed into the four corners of the
    # frame. So: size the canvas, then CopyI each bbox and Paste it into its corner.
    # The maker's reading is different: it slides each L inside the big grid and crops last.
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    ops, sels = [], []

    # background: the colour the canvas is painted with; the input is at least
    # twice the frame size in each dimension, so it is the majority colour
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # the four corner pieces: one per non-background colour
    cellmap = {}
    for r in range(fh):
        for c in range(fw):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))
    pieces = {}
    for col, cells in cellmap.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        s = set(cells)
        # each piece is an L; its bbox is missing only the corner opposite its elbow
        if (r1, c1) not in s:
            kind = 'UL'
        elif (r1, c0) not in s:
            kind = 'UR'
        elif (r0, c1) not in s:
            kind = 'LL'
        else:
            kind = 'LR'
        pieces[kind] = {'color': col, 'cells': s, 'r0': r0, 'c0': c0,
                        'h': r1 - r0 + 1, 'w': c1 - c0 + 1}
    KINDS = ['UL', 'UR', 'LL', 'LR']
    if not all(k in pieces for k in KINDS):
        ops.append(34); sels.append([0, 0, fh - 1, fw - 1])
        return ops, sels

    # frame size: stack the left-hand bboxes vertically and the top bboxes side by side
    h = pieces['UL']['h'] + pieces['LL']['h']
    w = pieces['UL']['w'] + pieces['UR']['w']
    # each bbox goes into the matching corner of the h x w frame
    dest = {
        'UL': (0, 0),
        'UR': (0, w - pieces['UR']['w']),
        'LL': (h - pieces['LL']['h'], 0),
        'LR': (h - pieces['LR']['h'], w - pieces['LR']['w']),
    }

    # 1. size the canvas to the frame (a transparent copy of I's top-left corner)
    G = np.zeros((h, w), dtype=int)
    G[:, :] = I[:h, :w]
    ops.append(33); sels.append([0, 0, h - 1, w - 1])

    # 2. copy each piece's bounding box from the input and paste it into its corner
    for k in KINDS:
        p = pieces[k]
        if int(p['color']) == 0:
            continue  # Copy/Paste cannot carry 0-valued cells; painted in step 3
        clip = I[p['r0']:p['r0'] + p['h'], p['c0']:p['c0'] + p['w']]
        dr, dc = dest[k]
        region = G[dr:dr + p['h'], dc:dc + p['w']]
        newreg = np.where(clip != 0, clip, region)
        if np.array_equal(newreg, region):
            continue
        ops.append(28); sels.append([p['r0'], p['c0'], p['h'] - 1, p['w'] - 1])
        ops.append(30); sels.append([dr, dc, 0, 0])
        G[dr:dr + p['h'], dc:dc + p['w']] = newreg

    # 3. a piece coloured 0 is invisible to Copy/Paste: paint the cells it lands on
    for k in KINDS:
        p = pieces[k]
        if int(p['color']) != 0:
            continue
        dr, dc = dest[k][0] - p['r0'], dest[k][1] - p['c0']
        land = [(r + dr, c + dc) for (r, c) in p['cells']]
        todo = [(r, c) for (r, c) in land if G[r, c] != 0]
        if todo:
            ops.append(0); sels.append(sel_of(todo))
            for r, c in todo:
                G[r, c] = 0

    # 4. the frame's interior is plain background: clean up whatever the canvas
    #    or a copied bbox left there (another piece inside a bbox, gaps between
    #    bboxes, 0-valued background that Paste does not carry)
    todo = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if G[r, c] != bgc]
    if todo:
        ops.append(bgc); sels.append(sel_of(todo))
        for r, c in todo:
            G[r, c] = bgc

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
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
                                f"for task a61ba2ce"
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
                    f"Failed to build a complete episode for task a61ba2ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a61ba2ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
