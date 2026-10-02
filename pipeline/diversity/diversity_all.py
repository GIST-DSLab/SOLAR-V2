#!/usr/bin/env python3
"""400개 전체에서 과제당 유효한 경로가 몇 가지인가.

좋고 나쁨이 아니라 다양성을 센다. 같은 인스턴스를 전부 푸는 코드들이 서로
다른 궤적을 내놓는지, 아니면 같은 것을 다시 쓴 것인지.

경로의 신원은 (op 이름, 실제로 고른 칸의 집합)의 나열이다. 선택을 bbox로
적든 칸 목록으로 적든 같은 칸을 고르면 같은 선택으로 센다 -- 표기 차이로
"다르다"가 나오면 전부 다르게 보인다.

과제 하나가 끝날 때마다 저장하고, 다시 돌리면 남은 것부터 간다. 쓰기는
임시 파일을 거쳐 바꿔치기하므로 도중에 끊겨도 파일이 깨지지 않는다.
"""
import sys, json, collections, os
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
OUT = Path("/hdd_data/yunho/cctmp/diversity_all.json")
DRAW = "/hdd_data/yunho/ARC_rearc_draw20"
N = 5

def save(rows):
    tmp = OUT.with_suffix(".part")
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(OUT)

def arms_for(t):
    return [d.name for d in sorted(ROOT.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and (d / t / "grid_maker.py").exists()]

def signature(derive, pairs, eps):
    out = []
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if g is None or g.shape != O.shape or not bool((g == O).all()):
            return None
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            rr, cc = np.where(m)
            steps.append((cm.NAME.get(int(op)),
                          frozenset(zip(rr.tolist(), cc.tolist()))))
        out.append(tuple(steps))
    return tuple(out)

def relation(a, b):
    if a == b:
        return "same"
    if len(a) != len(b):
        return "different"
    for x, y in zip(a, b):
        if collections.Counter(x) != collections.Counter(y):
            return "different"
    return "order"

tasks = sorted(p.name for p in (ROOT / "arc-agi-1").iterdir()
               if p.is_dir() and p.name != "__pycache__")
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {r["task"] for r in rows}
todo = [t for t in tasks if t not in done]
print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to go", flush=True)

for n, t in enumerate(todo, 1):
    try:
        pairs = cm.draw(t, N, [0])
    except Exception:
        pairs = None
    if not pairs:
        rows.append({"task": t, "valid": 0, "routes": 0, "arms": {}})
        save(rows); continue
    pairs = pairs[:N]
    eps = cm.episodes_from_draw(DRAW, t) or []
    sigs = {}
    for a in arms_for(t):
        try:
            d = cm.load_derive(ROOT / a / t / "grid_maker.py")
        except Exception:
            continue
        if d is None:
            continue
        try:
            s = signature(d, pairs, eps)
        except Exception:
            s = None
        if s is not None:
            sigs[a] = s
    groups = []
    for a, s in sigs.items():
        for g in groups:
            if relation(sigs[g[0]], s) == "same":
                g.append(a); break
        else:
            groups.append([a])
    reps = [g[0] for g in groups]
    order = diff = 0
    for i in range(1, len(reps)):
        r = relation(sigs[reps[0]], sigs[reps[i]])
        if r == "order": order += 1
        else: diff += 1
    rows.append({"task": t, "valid": len(sigs), "routes": len(groups),
                 "order": order, "diff": diff,
                 "arms": {a: i for i, g in enumerate(groups) for a in g}})
    save(rows)
    if n % 10 == 0:
        print(f"[{n}/{len(todo)}] {t}  푸는코드 {len(sigs)} 경로 {len(groups)}",
              flush=True)

print("done", len(rows), flush=True)
