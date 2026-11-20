"""Compose the Step 3 Phase 2 synthesis dashboard from the four published figures.

The four source panels are rasterised at 300 dpi by ``render_step3_phase2.py``.
This module places them on a shared 2x2 canvas and preserves their aspect ratios.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

LOGGER = logging.getLogger('step3_dashboard')

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES = ROOT / 'reports' / 'step3_phase2' / 'figures'
DEFAULT_OUTPUT = DEFAULT_FIGURES / 'step3_no_leakage_synthesis_dashboard.png'
MIN_FILE_BYTES = 400_000

INK = '#284768'
MUTED = '#5d6b76'
RULE = '#c3ccd3'
WHITE = '#ffffff'

PANELS = (
    'model_performance.png',
    'treeshap_physical.png',
    'module_attribution.png',
    'ale_supported_ranges.png',
)

# Layout geometry in inches.
MARGIN_X = 0.38
GAP_X = 0.25
HEADER_HEIGHT = 0.97
FOOTER_HEIGHT = 0.17
GAP_Y = 0.12
TOP_ROW_SHARE = 0.535


def configure_logging() -> None:
    """Route compact, tag-prefixed records to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def load_panels(figures_dir: Path) -> list[dict]:
    """Load and measure every source panel."""
    panels = []
    for filename in PANELS:
        path = figures_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing source figure: {path}')
        with Image.open(path) as handle:
            pixels = handle.size
            array = np.asarray(handle.convert('RGB'))
        if min(pixels) < 200:
            raise ValueError(f'Source figure is too small to embed: {path}')
        if not np.any(array < 248):
            raise ValueError(f'Source figure is blank: {path}')
        panels.append({'array': array, 'pixels': pixels})
    return panels


def compose(panels: list[dict], figsize: tuple[float, float]) -> plt.Figure:
    """Draw the 2x2 canvas with harmonised panel scales."""
    width, height = figsize
    grid_height = height - HEADER_HEIGHT - FOOTER_HEIGHT - GAP_Y
    row_heights = (grid_height * TOP_ROW_SHARE, grid_height * (1 - TOP_ROW_SHARE))
    column_width = (width - 2 * MARGIN_X - GAP_X) / 2
    if min(grid_height, column_width) <= 0:
        raise ValueError('Canvas is too small for the panel grid')

    figure = plt.figure(figsize=figsize, facecolor=WHITE)
    figure.text(MARGIN_X / width, 1 - 0.19 / height, 'Step 3 Phase 2 | No-leakage synthesis',
                ha='left', va='top', fontsize=20, fontweight='bold', color=INK)
    figure.text(MARGIN_X / width, 1 - 0.58 / height,
                'Family-isolated development validation and frozen November replay',
                ha='left', va='top', fontsize=10, color=MUTED)
    figure.add_artist(Line2D([MARGIN_X / width, 1 - MARGIN_X / width], [1 - 0.82 / height] * 2,
                             color=RULE, linewidth=0.9, transform=figure.transFigure))

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        cell_left = MARGIN_X + column * (column_width + GAP_X)
        cell_top = height - HEADER_HEIGHT - row * (row_heights[0] + GAP_Y)
        pixels_wide, pixels_high = panel['pixels']
        scale = min(column_width / pixels_wide, row_heights[row] / pixels_high)
        drawn_width = pixels_wide * scale
        drawn_height = pixels_high * scale
        placed_left = cell_left + (column_width - drawn_width) / 2
        placed_bottom = cell_top - drawn_height
        axes = figure.add_axes([placed_left / width, placed_bottom / height,
                                drawn_width / width, drawn_height / height])
        axes.imshow(panel['array'], interpolation='lanczos')
        axes.set_axis_off()

    return figure


def export(figure: plt.Figure, output: Path, dpi: int) -> None:
    """Write the dashboard at the requested resolution as an opaque RGB image."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=WHITE, format='png')
    plt.close(figure)
    with Image.open(output) as handle:
        opaque = handle.convert('RGB')
    opaque.save(output, format='png', dpi=(dpi, dpi), optimize=True)


def verify(output: Path, figsize: tuple[float, float], dpi: int) -> tuple[int, int]:
    """Check dimensions, resolution metadata and payload size of the export."""
    if not output.is_file():
        raise FileNotFoundError(f'Dashboard was not written: {output}')
    with Image.open(output) as handle:
        size = handle.size
        dpi_info = handle.info.get('dpi')
    expected = (round(figsize[0] * dpi), round(figsize[1] * dpi))
    if size != expected:
        raise ValueError(f'Unexpected canvas size {size}, expected {expected}')
    if dpi_info and any(abs(value - dpi) > 1 for value in dpi_info):
        raise ValueError(f'Unexpected resolution metadata: {dpi_info}')
    if output.stat().st_size < MIN_FILE_BYTES:
        raise ValueError(f'Dashboard payload too small: {output.stat().st_size} bytes')
    return size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-dir', type=Path, default=DEFAULT_FIGURES,
                        help='Directory holding the four source panels.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Destination path for the composite dashboard.')
    parser.add_argument('--dpi', type=int, default=300, help='Export resolution.')
    parser.add_argument('--width', type=float, default=17.0, help='Canvas width in inches.')
    parser.add_argument('--height', type=float, default=8.8, help='Canvas height in inches.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    figsize = (args.width, args.height)
    try:
        panels = load_panels(args.figures_dir)
        LOGGER.info('[INFO] Loaded %d source panels from %s', len(panels), args.figures_dir)
        figure = compose(panels, figsize)
        export(figure, args.output, args.dpi)
        size = verify(args.output, figsize, args.dpi)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.error('[FAIL] %s', error)
        return 1
    payload = args.output.stat().st_size / 1024 ** 2
    LOGGER.info('[PASS] Dashboard %d x %d px, %.2f MB, %d dpi', *size, payload, args.dpi)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
