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

    python viz/build_handcraft_gallery.py --root <data_folder>/whole \
                                          --out handcraft_gallery.html
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


def collect(root: Path, keep: tuple[str, ...]) -> list[dict]:
    tasks: dict[str, list[dict]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or "." not in folder.name:
            continue
        name = folder.name.split(".")[1]
        base, _, suffix = name.partition("-")
        if not suffix or not any(k in suffix for k in keep):
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


HTML = r"""<style>
/* The ARC palette is the content and it is loud: saturated magenta, orange,
   cyan, all fixed by the format. So the chrome recedes. Neutrals carry a slight
   blue bias toward the accent; the accent itself is teal and the secondary is
   violet, the two hues ARC's ten colours do not contain, so a label can never
   be mistaken for a grid cell. */
:root{
  --ground:#f6f8fa; --panel:#ffffff; --ink:#12171d; --muted:#5d6874;
  --line:#e3e8ee; --hair:#eef1f5;
  --expert:#0f7a82; --half:#5b4bb0;
  --chip:#eef2f6; --rule:rgba(0,0,0,.32);
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0e1116; --panel:#151a21; --ink:#e4e9f0; --muted:#828d9b;
    --line:#232b36; --hair:#1b222b;
    --expert:#45b6be; --half:#a396e8;
    --chip:#1b232e; --rule:rgba(255,255,255,.30);
  }
}
:root[data-theme="light"]{
  --ground:#f6f8fa; --panel:#ffffff; --ink:#12171d; --muted:#5d6874;
  --line:#e3e8ee; --hair:#eef1f5;
  --expert:#0f7a82; --half:#5b4bb0;
  --chip:#eef2f6; --rule:rgba(0,0,0,.32);
}
:root[data-theme="dark"]{
  --ground:#0e1116; --panel:#151a21; --ink:#e4e9f0; --muted:#828d9b;
  --line:#232b36; --hair:#1b222b;
  --expert:#45b6be; --half:#a396e8;
  --chip:#1b232e; --rule:rgba(255,255,255,.30);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
/* task ids and operation names are identifiers, so they are set in mono */
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:40px 32px 26px;max-width:1180px;margin:0 auto}
h1{margin:0 0 6px;font-size:27px;line-height:1.2;letter-spacing:-.3px;text-wrap:balance}
header p{margin:0;color:var(--muted);font-size:14px;max-width:65ch}
main{max-width:1180px;margin:0 auto;padding:0 32px 64px;
  display:flex;flex-direction:column;gap:20px}
.task{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:22px 24px;display:flex;flex-direction:column;gap:20px}
.task > h2{margin:0;font-size:16px;font-weight:600;letter-spacing:.5px}
.row{display:flex;flex-direction:column;gap:11px;padding-bottom:20px;
  border-bottom:1px solid var(--hair)}
.row:last-child{border-bottom:none;padding-bottom:0}
.tag{align-self:flex-start;padding:4px 11px;border-radius:999px;font-size:12.5px;
  font-weight:600;background:var(--chip);color:var(--expert)}
.tag.half{color:var(--half)}
.tag .n{color:var(--muted);font-weight:400}
.demos,.strip{display:flex;gap:13px;flex-wrap:wrap;align-items:flex-start;
  overflow-x:auto}
.demos{align-items:center;gap:16px}
.pair{display:flex;gap:6px;align-items:center}
.lab{color:var(--muted);font-size:11px;letter-spacing:.7px;text-transform:uppercase}
.frame{display:flex;flex-direction:column;align-items:center;gap:5px}
.frame .op{font-size:11.5px;color:var(--muted);max-width:118px;text-align:center}
.frame.last .op{color:var(--ink)}
canvas{display:block;image-rendering:pixelated;border-radius:2px}
.arrow{color:var(--muted);font-size:14px}
</style>

<header>
  <h1>__TITLE__</h1>
  <p>The same ARC task, solved by more than one hand-written route. Each frame is
     the grid an action was chosen from, outlined with the cells that action
     applies to. Variants of a task sit on adjacent rows so the routes can be
     read against each other.</p>
  <p id="sub" style="margin-top:8px"></p>
</header>
<main id="app"></main>

<script>
const DATA = __DATA__;
const P = __PALETTE__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function paint(cv, rows, cell, sel){
  const h = rows.length, w = rows[0].length;
  cv.width = w*cell+1; cv.height = h*cell+1;
  const g = cv.getContext("2d");
  g.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<h;r++) for(let c=0;c<w;c++){
    g.fillStyle = P[+rows[r][c]] || "#F0F0F0";
    g.fillRect(c*cell, r*cell, cell, cell);
  }
  // gridlines follow the theme: a white lattice vanishes on a light ground
  if(cell >= 5){
    g.strokeStyle = css("--rule"); g.lineWidth = 1; g.beginPath();
    for(let c=0;c<=w;c++){ g.moveTo(c*cell+.5,0); g.lineTo(c*cell+.5,h*cell); }
    for(let r=0;r<=h;r++){ g.moveTo(0,r*cell+.5); g.lineTo(w*cell,r*cell+.5); }
    g.stroke();
  }
  if(sel && sel.length){
    // the border of the region only. Outlining each selected cell turns a
    // multi-cell selection into a hatch, and Submit selects the whole grid.
    const on = new Set(sel.map(([r,c]) => r*100+c));
    const has = (r,c) => on.has(r*100+c);
    const seg = [];
    for(const [r,c] of sel){
      if(!has(r-1,c)) seg.push([c*cell,r*cell,(c+1)*cell,r*cell]);
      if(!has(r+1,c)) seg.push([c*cell,(r+1)*cell,(c+1)*cell,(r+1)*cell]);
      if(!has(r,c-1)) seg.push([c*cell,r*cell,c*cell,(r+1)*cell]);
      if(!has(r,c+1)) seg.push([(c+1)*cell,r*cell,(c+1)*cell,(r+1)*cell]);
    }
    for(const [col,wid] of [["#0d0d0d",4.6],["#ffffff",2.2]]){
      g.strokeStyle = col; g.lineWidth = wid; g.lineCap = "round"; g.beginPath();
      for(const [x0,y0,x1,y1] of seg){ g.moveTo(x0,y0); g.lineTo(x1,y1); }
      g.stroke();
    }
  }
}
const fit = (rows, budget) =>
  Math.max(3, Math.min(22, Math.floor(budget / Math.max(rows.length, rows[0].length))));

const jobs = [];                       // redrawn when the theme changes
function grid(rows, budget, sel){
  const cv = document.createElement("canvas");
  const cell = fit(rows, budget);
  jobs.push(() => paint(cv, rows, cell, sel));
  paint(cv, rows, cell, sel);
  return cv;
}
const arrow = () => {
  const a = document.createElement("span");
  a.className = "arrow"; a.textContent = "\u2192"; return a;
};

const app = document.getElementById("app");
let nvar = 0;
for(const t of DATA){
  nvar += t.variants.length;
  const sec = document.createElement("section"); sec.className = "task";
  const h2 = document.createElement("h2");
  h2.className = "mono"; h2.textContent = t.task; sec.appendChild(h2);

  for(const v of t.variants){
    const row = document.createElement("div"); row.className = "row";
    const tag = document.createElement("span");
    tag.className = "tag mono" + (v.variant.startsWith("half") ? " half" : "");
    tag.innerHTML = v.variant + ' <span class="n">' + (v.steps.length - 1) +
                    " actions, " + v.n_rollouts + " rollouts</span>";
    row.appendChild(tag);

    // Demonstrations belong to the variant, not the task: each variant samples
    // its own grids, so one variant's examples do not describe its siblings.
    const demos = document.createElement("div"); demos.className = "demos";
    const dl = document.createElement("span"); dl.className = "lab";
    dl.textContent = "demos"; demos.appendChild(dl);
    for(const ex of v.examples){
      const pair = document.createElement("div"); pair.className = "pair";
      pair.append(grid(ex.in, 72), arrow(), grid(ex.out, 72));
      demos.appendChild(pair);
    }
    row.appendChild(demos);

    const strip = document.createElement("div"); strip.className = "strip";
    v.steps.forEach((s, i) => {
      const f = document.createElement("div");
      f.className = "frame" + (s.op ? "" : " last");
      f.appendChild(grid(s.g, 108, s.s));
      const op = document.createElement("div");
      op.className = "op" + (s.op ? " mono" : "");
      op.textContent = s.op ? (i+1) + ". " + s.op : "result";
      f.appendChild(op); strip.appendChild(f);
    });
    row.appendChild(strip);
    sec.appendChild(row);
  }
  app.appendChild(sec);
}
document.getElementById("sub").textContent =
  nvar + " variants over " + DATA.length + " tasks, one validated rollout each.";

const redraw = () => jobs.forEach(f => f());
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement,
  {attributes:true, attributeFilter:["data-theme"]});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="a rollout of the handcraft maker set")
    ap.add_argument("--out", default="handcraft_gallery.html")
    ap.add_argument("--variants", nargs="+", default=["expert", "half"],
                    help="substrings a variant suffix must carry to be included")
    args = ap.parse_args()

    data = collect(Path(args.root), tuple(args.variants))
    title = "Handcraft: " + " and ".join(args.variants)
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__PALETTE__", json.dumps(PALETTE))
                .replace("__TITLE__", title))
    out = Path(args.out)
    out.write_text(html)
    n = sum(len(t["variants"]) for t in data)
    print(f"{n} variants over {len(data)} tasks -> {out} ({out.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
