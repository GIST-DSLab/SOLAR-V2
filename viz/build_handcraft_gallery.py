#!/usr/bin/env python3
"""Render the handcraft expert/half variants as one page, grouped by task.

The point of these variants is that the same ARC task is solved by more than one
hand-written route, so the useful view is not one trajectory at a time but every
variant of a task on adjacent rows. That is what this builds: one card per base
task, one filmstrip per variant, each frame showing the state an action was
chosen from with that action's selection outlined.

Only variants whose name carries `expert` or `half` are included, and only those
that actually produced a validated rollout — a maker that emits nothing but
`Submit` has nothing to look at.

    python viz/build_handcraft_gallery.py --root /hdd_data/yunho/ARC_handcraft_all/whole \
                                          --out /tmp/handcraft_gallery.html
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
           "#F012BE", "#FF851B", "#7FDBFF", "#870C25", "#F0F0F0"]


def enc(grid, dim) -> list[str]:
    """One grid as a list of digit strings, cropped to its real extent."""
    h, w = dim
    return ["".join(str(int(v)) for v in row[:w]) for row in grid[:h]]


def sel_cells(mask, dim) -> list[list[int]]:
    h, w = dim
    return [[r, c] for r in range(h) for c in range(w) if mask[r][c]]


def episode(path: Path) -> dict:
    d = json.loads(path.read_text())
    gd, grids = d["grid_dim"], d["grid"]
    names, smask = d.get("operation_name", []), d.get("selection_mask", [])
    steps = []
    for i in range(len(grids)):
        op = names[i] if i < len(names) else ""      # the terminal state has none
        steps.append({"op": op, "g": enc(grids[i], gd[i]),
                      "s": sel_cells(smask[i], gd[i]) if op and i < len(smask) else []})
    exs = [{"in": enc(ei, di), "out": enc(eo, do)}
           for ei, eo, di, do in zip(d.get("ex_in", []), d.get("ex_out", []),
                                     d.get("ex_in_grid_dim", []), d.get("ex_out_grid_dim", []))]
    return {"examples": exs[:3], "steps": steps,
            "in": enc(d["in_grid"], gd[0]), "out": enc(d["out_grid"], gd[-1])}


def collect(root: Path) -> list[dict]:
    tasks: dict[str, list[dict]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or "." not in folder.name:
            continue
        name = folder.name.split(".")[1]
        base, _, suffix = name.partition("-")
        if not suffix or ("expert" not in suffix and "half" not in suffix):
            continue
        files = sorted(folder.glob("*.json"), key=os.path.getsize)
        if not files:                       # nothing survived validation
            continue
        ep = episode(files[0])
        ep["variant"] = suffix
        ep["n_rollouts"] = len(files)
        tasks.setdefault(base, []).append(ep)
    order = {"expert": 0, "half": 1}
    for v in tasks.values():
        v.sort(key=lambda e: (order.get(e["variant"].split("-")[0], 2), e["variant"]))
    return [{"task": t, "variants": tasks[t]} for t in sorted(tasks)]


HTML = r"""<meta charset="utf-8">
<title>Handcraft — expert / half</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0c1015;color:#e6ebf2;font:14px/1.5 system-ui,-apple-system,sans-serif}
header{padding:28px 32px 20px;border-bottom:1px solid #242e3c}
h1{margin:0 0 4px;font-size:24px;letter-spacing:-.2px}
header p{margin:0;color:#8a94a5;font-size:13px}
.task{padding:26px 32px;border-bottom:1px solid #1a2230}
.task > h2{margin:0 0 14px;font:600 17px ui-monospace,monospace;letter-spacing:.4px}
.demos{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
.demos .pair{display:flex;gap:6px;align-items:center}
.lab{color:#5a6473;font-size:11px;letter-spacing:.3px;text-transform:uppercase}
.row{margin-bottom:22px;padding-bottom:18px;border-bottom:1px dashed #1f2735}
.row:last-child{border-bottom:none;padding-bottom:0}
.tag{display:inline-block;padding:3px 10px;border-radius:999px;font:600 12px ui-monospace,monospace;
  background:#1b232e;color:#40b8bf;margin-bottom:8px}
.tag.half{color:#e0a648}
.strip{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
.frame{text-align:center}
.frame .op{margin-top:5px;font-size:11px;color:#8a94a5;max-width:120px}
canvas{display:block;image-rendering:pixelated}
.arrow{color:#5a6473;font-size:15px}
</style>
<body>
<header>
  <h1>Handcraft — expert / half</h1>
  <p id="sub"></p>
</header>
<div id="app"></div>
<script>
const DATA = __DATA__;
const P = __PALETTE__;

function draw(rows, cell, sel){
  const h = rows.length, w = rows[0].length;
  const cv = document.createElement("canvas");
  cv.width = w*cell+1; cv.height = h*cell+1;
  const g = cv.getContext("2d");
  for(let r=0;r<h;r++) for(let c=0;c<w;c++){
    g.fillStyle = P[+rows[r][c]] || "#F0F0F0";
    g.fillRect(c*cell, r*cell, cell, cell);
  }
  if(cell >= 5){
    g.strokeStyle = "rgba(255,255,255,.28)"; g.lineWidth = 1; g.beginPath();
    for(let c=0;c<=w;c++){ g.moveTo(c*cell+.5,0); g.lineTo(c*cell+.5,h*cell); }
    for(let r=0;r<=h;r++){ g.moveTo(0,r*cell+.5); g.lineTo(w*cell,r*cell+.5); }
    g.stroke();
  }
  if(sel && sel.length){
    // border of the selected region only: outlining each cell turns a
    // multi-cell selection into a hatch, and Submit selects the whole grid
    const on = new Set(sel.map(([r,c]) => r*100+c));
    const has = (r,c) => on.has(r*100+c);
    const seg = [];
    for(const [r,c] of sel){
      if(!has(r-1,c)) seg.push([c*cell,r*cell,(c+1)*cell,r*cell]);
      if(!has(r+1,c)) seg.push([c*cell,(r+1)*cell,(c+1)*cell,(r+1)*cell]);
      if(!has(r,c-1)) seg.push([c*cell,r*cell,c*cell,(r+1)*cell]);
      if(!has(r,c+1)) seg.push([(c+1)*cell,r*cell,(c+1)*cell,(r+1)*cell]);
    }
    for(const [col,wid] of [["#101010",4.5],["#ffffff",2.2]]){
      g.strokeStyle = col; g.lineWidth = wid; g.lineCap = "round"; g.beginPath();
      for(const [x0,y0,x1,y1] of seg){ g.moveTo(x0,y0); g.lineTo(x1,y1); }
      g.stroke();
    }
  }
  return cv;
}
const fit = (rows, budget) =>
  Math.max(3, Math.min(22, Math.floor(budget / Math.max(rows.length, rows[0].length))));

const app = document.getElementById("app");
let nvar = 0;
for(const t of DATA){
  nvar += t.variants.length;
  const sec = document.createElement("div"); sec.className = "task";
  const h2 = document.createElement("h2"); h2.textContent = t.task; sec.appendChild(h2);

  for(const v of t.variants){
    const row = document.createElement("div"); row.className = "row";
    const tag = document.createElement("span");
    tag.className = "tag" + (v.variant.startsWith("half") ? " half" : "");
    tag.textContent = v.variant + " · " + v.steps.length + " frames · " +
                      v.n_rollouts + " rollouts";
    row.appendChild(tag);

    // Each variant samples its own grids, so the demonstrations belong to the
    // variant, not to the task. Hoisting one variant's demos above all the rows
    // put a red-and-orange pair over a blue-and-magenta trajectory.
    const demos = document.createElement("div"); demos.className = "demos";
    const dl = document.createElement("div"); dl.className = "lab";
    dl.textContent = "demos"; demos.appendChild(dl);
    for(const ex of v.examples){
      const pair = document.createElement("div"); pair.className = "pair";
      pair.appendChild(draw(ex.in, fit(ex.in, 72)));
      const a = document.createElement("span"); a.className = "arrow"; a.textContent = "→";
      pair.appendChild(a);
      pair.appendChild(draw(ex.out, fit(ex.out, 72)));
      demos.appendChild(pair);
    }
    row.appendChild(demos);
    const strip = document.createElement("div"); strip.className = "strip";
    v.steps.forEach((s, i) => {
      const f = document.createElement("div"); f.className = "frame";
      f.appendChild(draw(s.g, fit(s.g, 108), s.s));
      const op = document.createElement("div"); op.className = "op";
      op.textContent = s.op ? (i+1) + ". " + s.op : "result";
      f.appendChild(op); strip.appendChild(f);
    });
    row.appendChild(strip); sec.appendChild(row);
  }
  app.appendChild(sec);
}
document.getElementById("sub").textContent =
  nvar + " variants over " + DATA.length + " tasks · one validated rollout each";
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hdd_data/yunho/ARC_handcraft_all/whole")
    ap.add_argument("--out", default="/tmp/handcraft_gallery.html")
    args = ap.parse_args()

    data = collect(Path(args.root))
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__PALETTE__", json.dumps(PALETTE)))
    out = Path(args.out)
    out.write_text(html)
    n = sum(len(t["variants"]) for t in data)
    print(f"{n} variants over {len(data)} tasks -> {out} ({out.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
