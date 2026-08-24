# Figures and gallery pages

Everything that turns a rollout into something you can look at. None of these
are needed to produce data.

The trajectory scripts take `--root <data_folder>/whole`, the directory
`pipeline/gen_rearc_trajectories_v2.py` writes. `build_hero_bg.py` is the
exception: it reads a `build_preview.py` parquet instead.

To browse episodes rather than render them, use the published viewer at
[gist-dslab.github.io/SOLAR-V2](https://gist-dslab.github.io/SOLAR-V2/), which needs
nothing from this directory.

---

## One trajectory

### `visualize_trajectory.py` / `visualize_trajectory_gif.py`

A single episode as a still figure or an animated GIF.

The figure keeps the original SOLAR-Generator layout: one row per demonstration
pair, input beside output, then the trajectory along the bottom — test input
first, then one panel per step, each labelled underneath with the op number and
name. Two differences: the selection an op applies to is outlined on its panel,
and grids are clipped to `grid_dim`, so the padding a 30x30 recording carries
is not drawn. The GIF in the top-level README is the other script's output.

| flag | default | |
|---|---|---|
| `--root` | required | a rollout directory |
| `--task` | figure: pick one; GIF: required | task id |
| `--file` | — | a specific episode JSON, instead of `--task` |
| `--out` | `figure/trajectory.png` / `figure/teaser.gif` | |
| `--max_steps` | 8 (figure), 12 (GIF) | frames to show |
| `--max_examples` / `--examples` | 3 (figure) / 2 (GIF) | worked examples drawn |
| `--min_steps` | 3 | figure: skip episodes shorter than this when picking |
| `--list` | off | figure: list candidate tasks and exit |
| `--title` | — | figure only |
| `--ms` | 900 | GIF: milliseconds per frame |
| `--hold` | 2600 | GIF: milliseconds on the final frame |
| `--dpi` | 110 | GIF only |

```bash
python viz/visualize_trajectory.py     --root $SOLAR_DATA_ROOT/draw0/whole --task 05f2a901 --out figure/mine.png
python viz/visualize_trajectory_gif.py --root $SOLAR_DATA_ROOT/draw0/whole --task 05f2a901 --out figure/mine.gif
```

---

## Pages

### `build_handcraft_gallery.py`

One page per task, with every hand-written variant of that task on adjacent
rows, so two routes to the same target sit side by side. Each variant carries
its own worked examples: the variants sample their own grids, so hoisting one
set of demonstrations over the whole task shows the wrong grids.

| flag | default | |
|---|---|---|
| `--root` | required | a rollout of the `handcraft` set |
| `--out` | `handcraft_gallery.html` | |
| `--variants` | `expert half` | which variant suffixes to include |

### `build_hero_bg.py`

The backdrop on the published overview page: a mosaic of real grids, bucketed
by dominant colour and dealt round-robin so no region of the image is
monochrome. Deliberately not blurred here, since the page blurs it in CSS.

| flag | default | |
|---|---|---|
| `--preview` | required | a `build_preview.py` parquet |
| `--out` | `docs/hero.png` | |
| `--width` / `--height` | 1920 / 1000 | |
| `--col` | 176 | tile column width in pixels |
| `--gap` | 20 | |
| `--seed` | 7 | |

---

## Rendering notes

Grids are drawn as flat colour blocks with **no cell separators**. Separators
read well at 1:1 and fall apart at every other scale: a 1px line survives
nearest-neighbour downsampling only where the sampling grid happens to land on
it, so a thumbnail comes out with a few thick white bands, most of the lattice
missing, and whole cell rows dropped. Flat blocks downscale to the right shape
at any factor.

Selections are outlined on the region border only. Outlining each selected cell
turns a multi-cell selection into a hatch, and `Submit` selects the whole grid,
which came out as a white mesh over the entire picture.
