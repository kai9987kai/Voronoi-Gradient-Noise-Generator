"""Deterministic Voronoi and gradient-noise texture generator."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, PngImagePlugin

VERSION = "2.0.0"
TAU = math.tau

Color = tuple[int, int, int]
Palette = tuple[tuple[float, Color], ...]

PALETTES: dict[str, Palette] = {
    "aurora": (
        (0.00, (8, 10, 35)),
        (0.24, (40, 25, 85)),
        (0.48, (21, 112, 130)),
        (0.72, (80, 210, 164)),
        (1.00, (239, 255, 203)),
    ),
    "ember": (
        (0.00, (15, 8, 20)),
        (0.28, (75, 14, 37)),
        (0.52, (184, 45, 29)),
        (0.76, (248, 135, 36)),
        (1.00, (255, 243, 164)),
    ),
    "gold": (
        (0.00, (8, 8, 7)),
        (0.32, (57, 43, 8)),
        (0.62, (179, 125, 12)),
        (0.82, (246, 204, 59)),
        (1.00, (255, 249, 193)),
    ),
    "ice": (
        (0.00, (7, 17, 31)),
        (0.30, (19, 65, 91)),
        (0.58, (48, 143, 166)),
        (0.80, (151, 224, 222)),
        (1.00, (244, 255, 255)),
    ),
    "mono": (
        (0.00, (5, 5, 7)),
        (0.45, (75, 78, 87)),
        (0.72, (174, 178, 185)),
        (1.00, (255, 255, 255)),
    ),
    "terrain": (
        (0.00, (6, 24, 52)),
        (0.30, (13, 74, 103)),
        (0.45, (28, 121, 100)),
        (0.58, (112, 139, 67)),
        (0.76, (129, 96, 62)),
        (0.90, (190, 181, 153)),
        (1.00, (250, 250, 245)),
    ),
}

STYLES = ("hybrid", "terrain", "cells", "marble", "ridges")


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for a generated texture."""

    width: int = 256
    height: int = 256
    seed: int = 42
    style: str = "hybrid"
    palette: str = "aurora"
    scale: float = 5.5
    cell_density: float = 14.0
    octaves: int = 5
    persistence: float = 0.52
    lacunarity: float = 2.0
    warp: float = 0.16
    angle_offset: float = 0.5
    contrast: float = 1.08
    lighting: float = 0.75

    def validate(self) -> None:
        if not 8 <= self.width <= 4096 or not 8 <= self.height <= 4096:
            raise ValueError("width and height must be between 8 and 4096")
        if self.style not in STYLES:
            raise ValueError(f"unknown style: {self.style}")
        if self.palette not in PALETTES:
            raise ValueError(f"unknown palette: {self.palette}")
        if self.scale <= 0 or self.cell_density <= 0:
            raise ValueError("scale and cell density must be positive")
        if not 1 <= self.octaves <= 10:
            raise ValueError("octaves must be between 1 and 10")
        if not 0 < self.persistence <= 1:
            raise ValueError("persistence must be greater than 0 and at most 1")
        if self.lacunarity < 1:
            raise ValueError("lacunarity must be at least 1")
        if not 0 <= self.warp <= 1:
            raise ValueError("warp must be between 0 and 1")
        if not 0.1 <= self.contrast <= 3:
            raise ValueError("contrast must be between 0.1 and 3")
        if not 0 <= self.lighting <= 2:
            raise ValueError("lighting must be between 0 and 2")


@dataclass(frozen=True)
class VoronoiSample:
    nearest: float
    second_nearest: float
    cell_value: float
    offset_x: float
    offset_y: float

    @property
    def edge(self) -> float:
        return self.second_nearest - self.nearest


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _smoothstep(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _fade(value: float) -> float:
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _hash_u32(x: int, y: int, seed: int) -> int:
    value = (x * 0x1F123BB5) ^ (y * 0x5F356495) ^ (seed * 0x6C8E9CF5)
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _random01(x: int, y: int, seed: int) -> float:
    return _hash_u32(x, y, seed) / 0xFFFFFFFF


@lru_cache(maxsize=65_536)
def _gradient(ix: int, iy: int, seed: int) -> tuple[float, float]:
    angle = _random01(ix, iy, seed) * TAU
    return math.cos(angle), math.sin(angle)


def _perlin(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = x - x0
    ty = y - y0

    g00 = _gradient(x0, y0, seed)
    g10 = _gradient(x0 + 1, y0, seed)
    g01 = _gradient(x0, y0 + 1, seed)
    g11 = _gradient(x0 + 1, y0 + 1, seed)

    n00 = g00[0] * tx + g00[1] * ty
    n10 = g10[0] * (tx - 1.0) + g10[1] * ty
    n01 = g01[0] * tx + g01[1] * (ty - 1.0)
    n11 = g11[0] * (tx - 1.0) + g11[1] * (ty - 1.0)

    u = _fade(tx)
    v = _fade(ty)
    raw = _lerp(_lerp(n00, n10, u), _lerp(n01, n11, u), v)
    return _clamp(0.5 + raw * 0.70710678118)


def gradient_noise(
    uv: Sequence[float], scale: float, seed: int = 0
) -> float:
    """Return coherent 2D gradient noise in the range 0..1."""

    return _perlin(float(uv[0]) * scale, float(uv[1]) * scale, seed)


def fractal_noise(
    uv: Sequence[float],
    scale: float,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> float:
    """Combine gradient-noise octaves into fractal Brownian motion."""

    if octaves < 1:
        raise ValueError("octaves must be at least 1")
    if scale <= 0 or persistence <= 0 or lacunarity < 1:
        raise ValueError("invalid fractal-noise parameters")

    total = 0.0
    amplitude = 1.0
    amplitude_sum = 0.0
    frequency = scale

    for octave in range(octaves):
        total += gradient_noise(uv, frequency, seed + octave * 1_013) * amplitude
        amplitude_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / amplitude_sum


@lru_cache(maxsize=65_536)
def _feature_point(
    cell_x: int, cell_y: int, seed: int, angle_offset: float
) -> tuple[float, float]:
    angle = _random01(cell_x, cell_y, seed) * TAU + angle_offset
    radius = 0.12 + _random01(cell_x, cell_y, seed + 97) * 0.36
    return 0.5 + math.cos(angle) * radius, 0.5 + math.sin(angle) * radius


def voronoi_random_vector(
    uv: Sequence[float], angle_offset: float, seed: int = 0
) -> list[float]:
    """Return the seeded feature point for a Voronoi lattice cell."""

    point = _feature_point(
        math.floor(float(uv[0])),
        math.floor(float(uv[1])),
        seed,
        angle_offset,
    )
    return [point[0], point[1]]


def _voronoi_sample(
    uv: Sequence[float],
    angle_offset: float,
    cell_density: float,
    seed: int,
) -> VoronoiSample:
    px = float(uv[0]) * cell_density
    py = float(uv[1]) * cell_density
    cell_x = math.floor(px)
    cell_y = math.floor(py)
    local_x = px - cell_x
    local_y = py - cell_y

    nearest = float("inf")
    second_nearest = float("inf")
    nearest_cell = (cell_x, cell_y)
    nearest_offset = (0.5, 0.5)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            candidate_x = cell_x + dx
            candidate_y = cell_y + dy
            offset = _feature_point(
                candidate_x, candidate_y, seed, angle_offset
            )
            delta_x = dx + offset[0] - local_x
            delta_y = dy + offset[1] - local_y
            distance = math.hypot(delta_x, delta_y)

            if distance < nearest:
                second_nearest = nearest
                nearest = distance
                nearest_cell = (candidate_x, candidate_y)
                nearest_offset = offset
            elif distance < second_nearest:
                second_nearest = distance

    return VoronoiSample(
        nearest=nearest,
        second_nearest=second_nearest,
        cell_value=_random01(nearest_cell[0], nearest_cell[1], seed + 211),
        offset_x=nearest_offset[0],
        offset_y=nearest_offset[1],
    )


def voronoi_noise(
    uv: Sequence[float],
    angle_offset: float,
    cell_density: float,
    seed: int = 0,
) -> list[float]:
    """Return nearest distance and feature-point offset for compatibility."""

    sample = _voronoi_sample(uv, angle_offset, cell_density, seed)
    return [sample.nearest, sample.offset_x, sample.offset_y]


def _domain_warp(uv: tuple[float, float], config: NoiseConfig) -> tuple[float, float]:
    if config.warp == 0:
        return uv

    warp_scale = max(0.5, config.scale * 0.55)
    warp_x = fractal_noise(
        (uv[0] + 17.13, uv[1] - 8.27),
        warp_scale,
        octaves=min(3, config.octaves),
        persistence=config.persistence,
        lacunarity=config.lacunarity,
        seed=config.seed + 10_007,
    )
    warp_y = fractal_noise(
        (uv[0] - 4.91, uv[1] + 23.71),
        warp_scale,
        octaves=min(3, config.octaves),
        persistence=config.persistence,
        lacunarity=config.lacunarity,
        seed=config.seed + 20_011,
    )
    return (
        uv[0] + (warp_x - 0.5) * config.warp,
        uv[1] + (warp_y - 0.5) * config.warp,
    )


def _sample_field(
    uv: tuple[float, float], config: NoiseConfig
) -> tuple[float, float]:
    warped = _domain_warp(uv, config)
    base = fractal_noise(
        warped,
        config.scale,
        config.octaves,
        config.persistence,
        config.lacunarity,
        config.seed,
    )
    detail = fractal_noise(
        (warped[0] + 31.7, warped[1] - 12.4),
        config.scale * 2.35,
        min(3, config.octaves),
        config.persistence,
        config.lacunarity,
        config.seed + 30_013,
    )
    voronoi = _voronoi_sample(
        warped,
        config.angle_offset,
        config.cell_density,
        config.seed + 40_009,
    )
    cell_center = 1.0 - _clamp(voronoi.nearest * 1.35)
    cell_edge = _smoothstep(voronoi.edge * 4.0)

    if config.style == "terrain":
        value = base * 0.82 + detail * 0.18
        structure = 1.0
    elif config.style == "cells":
        value = (
            voronoi.cell_value * 0.50
            + cell_center * 0.32
            + base * 0.18
        )
        structure = 0.48 + cell_edge * 0.52
    elif config.style == "marble":
        flow = warped[0] * config.scale + (base - 0.5) * 5.5
        value = 0.5 + 0.5 * math.sin(flow * TAU)
        value = value * 0.78 + detail * 0.22
        structure = 0.86 + cell_edge * 0.14
    elif config.style == "ridges":
        ridge = 1.0 - abs(base * 2.0 - 1.0)
        value = ridge * ridge * 0.78 + cell_center * 0.22
        structure = 0.78 + cell_edge * 0.22
    else:
        value = base * 0.64 + detail * 0.16 + cell_center * 0.20
        structure = 0.78 + cell_edge * 0.22

    return value, structure


def _palette_color(value: float, palette: Palette) -> Color:
    value = _clamp(value)
    for index in range(1, len(palette)):
        right_position, right_color = palette[index]
        if value <= right_position:
            left_position, left_color = palette[index - 1]
            span = right_position - left_position
            amount = 0.0 if span == 0 else (value - left_position) / span
            return tuple(
                round(_lerp(left_color[channel], right_color[channel], amount))
                for channel in range(3)
            )
    return palette[-1][1]


def _normalize(values: list[float], contrast: float) -> list[float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span < 1e-12:
        return [0.5] * len(values)

    normalized = [(value - low) / span for value in values]
    return [
        _smoothstep(_clamp((value - 0.5) * contrast + 0.5))
        for value in normalized
    ]


def render(config: NoiseConfig) -> Image.Image:
    """Render a configured texture as a Pillow RGB image."""

    config.validate()
    values: list[float] = []
    structures: list[float] = []
    x_denominator = max(1, config.width - 1)
    y_denominator = max(1, config.height - 1)

    for y in range(config.height):
        v = y / y_denominator
        for x in range(config.width):
            value, structure = _sample_field((x / x_denominator, v), config)
            values.append(value)
            structures.append(structure)

    values = _normalize(values, config.contrast)
    palette = PALETTES[config.palette]
    pixels: list[Color] = []

    for y in range(config.height):
        for x in range(config.width):
            index = y * config.width + x
            left = values[index - 1] if x > 0 else values[index]
            right = values[index + 1] if x + 1 < config.width else values[index]
            up = values[index - config.width] if y > 0 else values[index]
            down = (
                values[index + config.width]
                if y + 1 < config.height
                else values[index]
            )
            slope_x = right - left
            slope_y = down - up
            light = _clamp(
                0.88
                + config.lighting * (-slope_x * 1.8 - slope_y * 1.2),
                0.48,
                1.18,
            )
            shade = light * structures[index]
            color = _palette_color(values[index], palette)
            pixels.append(
                tuple(round(_clamp(channel * shade, 0, 255)) for channel in color)
            )

    image = Image.new("RGB", (config.width, config.height))
    image.putdata(pixels)
    return image


def save_image(image: Image.Image, output: Path, config: NoiseConfig) -> None:
    """Save an image, embedding generator settings when PNG is used."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Software", f"Voronoi Gradient Noise Generator {VERSION}")
        metadata.add_text("NoiseConfig", json.dumps(asdict(config), sort_keys=True))
        image.save(output, pnginfo=metadata)
    else:
        image.save(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic, domain-warped Voronoi textures."
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("noise.png"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style", choices=STYLES, default="hybrid")
    parser.add_argument("--palette", choices=tuple(PALETTES), default="aurora")
    parser.add_argument("--scale", type=float, default=5.5)
    parser.add_argument("--cells", dest="cell_density", type=float, default=14.0)
    parser.add_argument("--octaves", type=int, default=5)
    parser.add_argument("--persistence", type=float, default=0.52)
    parser.add_argument("--lacunarity", type=float, default=2.0)
    parser.add_argument("--warp", type=float, default=0.16)
    parser.add_argument("--angle", dest="angle_offset", type=float, default=0.5)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--lighting", type=float, default=0.75)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="open the generated image after saving it",
    )
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="print available palettes and exit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_palettes:
        print("\n".join(PALETTES))
        return 0

    config = NoiseConfig(
        width=args.width,
        height=args.height,
        seed=args.seed,
        style=args.style,
        palette=args.palette,
        scale=args.scale,
        cell_density=args.cell_density,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        warp=args.warp,
        angle_offset=args.angle_offset,
        contrast=args.contrast,
        lighting=args.lighting,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))

    image = render(config)
    save_image(image, args.output, config)
    print(
        f"Generated {args.output} "
        f"({config.width}x{config.height}, {config.style}, seed {config.seed})"
    )
    if args.preview:
        image.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
