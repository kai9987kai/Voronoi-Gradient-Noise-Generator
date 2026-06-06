import json

import pytest
from PIL import Image

from main import (
    NoiseConfig,
    fractal_noise,
    gradient_noise,
    render,
    save_image,
    voronoi_noise,
)


def test_noise_samples_are_deterministic_and_bounded():
    uv = (0.314, 0.159)

    gradient = gradient_noise(uv, scale=8.0, seed=12)
    assert gradient == gradient_noise(uv, scale=8.0, seed=12)
    assert 0.0 <= gradient <= 1.0

    fractal = fractal_noise(uv, scale=4.0, octaves=4, seed=12)
    assert fractal == fractal_noise(uv, scale=4.0, octaves=4, seed=12)
    assert 0.0 <= fractal <= 1.0

    voronoi = voronoi_noise(uv, angle_offset=0.5, cell_density=10, seed=12)
    assert voronoi == voronoi_noise(
        uv, angle_offset=0.5, cell_density=10, seed=12
    )
    assert voronoi[0] >= 0.0
    assert all(0.0 <= component <= 1.0 for component in voronoi[1:])


def test_fractal_noise_rejects_invalid_sampling_parameters():
    with pytest.raises(ValueError):
        fractal_noise((0.5, 0.5), scale=4.0, octaves=0)


def test_render_is_reproducible_and_seeded():
    config = NoiseConfig(width=24, height=20, seed=7, octaves=3)
    first = render(config)
    second = render(config)
    different = render(
        NoiseConfig(width=24, height=20, seed=8, octaves=3)
    )

    assert first.size == (24, 20)
    assert first.mode == "RGB"
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != different.tobytes()
    raw_pixels = first.tobytes()
    colors = {
        raw_pixels[index : index + 3]
        for index in range(0, len(raw_pixels), 3)
    }
    assert len(colors) > 20


@pytest.mark.parametrize(
    ("style", "palette"),
    [
        ("hybrid", "aurora"),
        ("terrain", "terrain"),
        ("cells", "gold"),
        ("marble", "ice"),
        ("ridges", "ember"),
    ],
)
def test_all_styles_render(style, palette):
    image = render(
        NoiseConfig(
            width=16,
            height=16,
            style=style,
            palette=palette,
            octaves=2,
        )
    )
    assert image.getbbox() == (0, 0, 16, 16)


def test_png_contains_generation_metadata(tmp_path):
    config = NoiseConfig(width=16, height=16, style="cells", octaves=2)
    output = tmp_path / "texture.png"
    save_image(render(config), output, config)

    with Image.open(output) as image:
        saved_config = json.loads(image.info["NoiseConfig"])
        assert saved_config["style"] == "cells"
        assert saved_config["seed"] == config.seed
        assert image.info["Software"].startswith(
            "Voronoi Gradient Noise Generator"
        )


@pytest.mark.parametrize(
    "config",
    [
        NoiseConfig(width=4),
        NoiseConfig(octaves=0),
        NoiseConfig(warp=1.1),
        NoiseConfig(style="unknown"),
        NoiseConfig(palette="unknown"),
    ],
)
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()
