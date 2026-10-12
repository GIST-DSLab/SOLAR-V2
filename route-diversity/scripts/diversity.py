"""한 과제에 코드가 여럿 있을 때, 유효한 경로가 몇 가지로 갈라지는가.

좋고 나쁨을 묻지 않는다. 묻는 것은 다양성이다: 같은 인스턴스를 전부 푸는
코드들이 서로 다른 궤적을 내놓는가, 아니면 같은 것을 다시 쓴 것인가.

경로의 신원은 (op 이름, 고른 칸의 집합)의 나열로 잡는다. 선택을 어떻게
적었는지는 상관없다 -- bbox로 적든 칸 목록으로 적든 같은 칸을 고르면 같은
선택이다. 셋으로 가른다.

  같음        단계별로 op과 고른 칸이 모두 같다
  순서만      같은 단계들의 순서만 다르다
  다른 일     그 밖 -- 다른 op을 쓰거나 같은 op을 다른 영역에 쓴다
"""
import sys, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
TASKS = sorted(p.name for p in (ROOT / "golf1").iterdir()
               if p.is_dir() and p.name != "__pycache__")
N = 5

def arms_for(t):
    out = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith(".") \
                and (d / t / "grid_maker.py").exists():
            out.append(d.name)
    return out

def signature(derive, pairs, eps):
    """이 코드가 다섯 인스턴스에 내놓는 궤적. 하나라도 못 풀면 None."""
    steps_all = []
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if g is None or g.shape != O.shape or not bool((g == O).all()):
            return None
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            rr, cc = np.where(m)
            steps.append((cm.NAME.get(int(op)), frozenset(zip(rr.tolist(), cc.tolist()))))
        steps_all.append(tuple(steps))
    return tuple(steps_all)

def relation(a, b):
    if a == b:
        return "same"
    if len(a) != len(b):
        return "different"
    for x, y in zip(a, b):
        if collections.Counter(x) != collections.Counter(y):
            return "different"
    return "order"

rows = []
for t in TASKS:
    pairs = cm.draw(t, N, [0])
    if not pairs:
        continue
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    sigs = {}
    for a in arms_for(t):
        d = cm.load_derive(ROOT / a / t / "grid_maker.py")
        if d is None:
            continue
        s = signature(d, pairs, eps)
        if s is not None:
            sigs[a] = s
    if len(sigs) < 2:
        rows.append({"task": t, "valid": len(sigs), "routes": len(sigs),
                     "order": 0, "diff": max(0, len(sigs) - 1)})
        continue
    # 서로 다른 경로끼리 묶는다
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

json.dump(rows, open("/hdd_data/yunho/cctmp/diversity.json", "w"))
print(f"{'task':10s} {'푸는코드':>7s} {'다른경로':>8s} {'순서만':>6s} {'다른일':>6s}")
for r in rows:
    print(f"{r['task']:10s} {r['valid']:7d} {r['routes']:8d} {r['order']:6d} {r['diff']:6d}")
v = [r for r in rows if r["valid"] >= 2]
print(f"\n코드가 2개 이상 통하는 과제 {len(v)}/{len(rows)}")
print(f"평균: 푸는 코드 {sum(r['valid'] for r in v)/len(v):.1f}개 -> "
      f"서로 다른 경로 {sum(r['routes'] for r in v)/len(v):.1f}개")
print(f"경로가 1가지뿐인 과제 {sum(1 for r in v if r['routes']==1)}")
print(f"경로가 3가지 이상인 과제 {sum(1 for r in v if r['routes']>=3)}")
