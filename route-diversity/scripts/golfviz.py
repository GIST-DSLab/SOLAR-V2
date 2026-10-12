"""다섯 과제의 두 궤적을 같은 입력으로 뽑는다 (배포본 / golf1)."""
import sys, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path
import gymnasium as gym

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
TASKS = ["264363fd", "ce4f8723", "c3e719e8", "d0f5fe59", "c8cbb738"]

def trace(derive, I, O, shown):
    """각 스텝의 op, 선택 마스크, 결과 격자."""
    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=cm._One(I, O), max_grid_size=cm.MAX_GRID_DIM,
                   colors=10, max_episode_steps=None, max_trial=1)
    try:
        ops, sels = cm.call_derive(derive, I.tolist(), O.tolist(), shown)
        obs, _ = env.reset(options={"prob_index": 0, "adaptation": False})
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            obs, _, _, _, _ = env.step({"selection": m, "operation": int(op)})
            h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
            g = np.asarray(obs["grid"])[:h, :w].astype(int)
            rr, cc = np.where(m)
            if len(rr):
                box = (int(rr.max()-rr.min()+1)) * (int(cc.max()-cc.min()+1))
                arb = bool(int(m.sum()) < box)
                cells = [[int(a), int(b)] for a, b in zip(rr, cc)]
            else:
                arb, cells = False, []
            steps.append({"op": cm.NAME.get(int(op)), "arb": arb,
                          "sel": cells, "grid": g.tolist()})
        return steps
    except Exception as e:
        return None
    finally:
        env.close()

out = []
for t in TASKS:
    pairs = cm.draw(t, 10, [0])
    if not pairs:
        continue
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    # 두 쪽 다 푸는 첫 인스턴스
    for I, O in pairs:
        shown = eps[0][0] if eps else None
        got = {}
        for arm, d in (("release", ROOT/"arc-agi-1"/t/"grid_maker.py"),
                       ("golf", ROOT/"golf1"/t/"grid_maker.py")):
            fn = cm.load_derive(d)
            st = trace(fn, I, O, shown) if fn else None
            if st and np.array_equal(np.array(st[-1]["grid"]), O):
                got[arm] = st
        if len(got) == 2:
            out.append({"task": t, "input": I.tolist(), "output": O.tolist(),
                        **got})
            break

json.dump(out, open("/hdd_data/yunho/cctmp/golfviz.json", "w"))
for r in out:
    print(r["task"], "release", len(r["release"]), "ops /", "golf", len(r["golf"]), "ops",
          " 임의마스크", sum(s["arb"] for s in r["release"]), "vs", sum(s["arb"] for s in r["golf"]))
