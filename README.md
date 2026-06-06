# Voronoi Gradient Noise Generator

A deterministic procedural texture generator that combines coherent gradient
noise, fractal Brownian motion, Voronoi cells, domain warping, palette mapping,
and lightweight surface shading.

The original prototype used independent random values for every pixel. Version
2 replaces that static with continuous, reproducible noise and adds a practical
command-line and Python API.

## Features

- Seeded output: the same settings always produce the same image
- Five styles: `hybrid`, `terrain`, `cells`, `marble`, and `ridges`
- Six built-in palettes
- Multi-octave gradient noise with adjustable persistence and lacunarity
- Organic domain warping and Voronoi edge detail
- Automatic contrast normalization and directional lighting
- Embedded generation settings in PNG metadata
- No runtime dependency beyond Pillow

## Install

```powershell
python -m pip install -r requirements.txt
```

## Generate

```powershell
python main.py
python main.py --style cells --palette gold --seed 2026 -o cells.png
python main.py --style terrain --palette terrain --width 1024 --height 1024 -o terrain.png
python main.py --style marble --palette ice --scale 3.5 --warp 0.3 -o marble.png
```

Use `python main.py --help` for every option. Add `--preview` to open the image
after it is saved. The fast default is 256×256; raise `--width` and `--height`
for final high-resolution exports.

Important controls:

| Option | Effect |
| --- | --- |
| `--seed` | Selects a reproducible variation |
| `--scale` | Controls the size of gradient-noise features |
| `--cells` | Controls Voronoi cell density |
| `--octaves` | Adds progressively finer detail |
| `--warp` | Distorts coordinates for more organic forms |
| `--contrast` | Expands or compresses tonal separation |
| `--lighting` | Controls relief shading strength |

## Python API

```python
from pathlib import Path

from main import NoiseConfig, render, save_image

config = NoiseConfig(
    width=768,
    height=768,
    seed=17,
    style="ridges",
    palette="ember",
    warp=0.24,
)
image = render(config)
save_image(image, Path("ridges.png"), config)
```

`gradient_noise`, `fractal_noise`, `voronoi_random_vector`, and
`voronoi_noise` are also available for direct sampling.

## Test

```powershell
python -m pytest
```
