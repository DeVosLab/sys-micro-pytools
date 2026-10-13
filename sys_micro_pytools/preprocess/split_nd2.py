from pathlib import Path
import re
from typing import Sequence, Union

import numpy as np
import tifffile
from nd2reader import ND2Reader
from tqdm import tqdm


# Match a Nikon NIS-Elements 'Well<row><col>' token, e.g. 'WellB02', 'WellH12'.
WELL_RE = re.compile(r'(Well)([A-Za-z]\d{1,3})')

# Match an existing 'Point<row><col>_<index>' token, e.g. 'PointB02_0000'.
POINT_RE = re.compile(r'Point[A-Za-z]\d{1,3}_\d+')


def _build_output_stem(stem: str, fov_index: int, well_code: Union[str, None]) -> str:
    """Build the output filename stem for a given FOV (point) index.

    The output follows the Nikon NIS-Elements naming convention:
    ``Well<code>_Point<code>_<NNNN>_...`` where ``NNNN`` is the zero-padded
    FOV index.

    - If ``stem`` already contains a ``Point<code>_<NNNN>`` token (i.e. the
      input file was already split per point), the existing index is updated
      to ``fov_index``.
    - Otherwise, ``Point<code>_<NNNN>`` is inserted right after the
      ``Well<code>`` token. If no ``Well<code>`` token is found, the point
      token is appended to the stem.
    """
    if POINT_RE.search(stem):
        return POINT_RE.sub(
            lambda m: re.sub(r'_\d+$', f'_{fov_index:04d}', m.group(0)),
            stem,
            count=1,
        )

    if well_code is None:
        return f"{stem}_Point_{fov_index:04d}"

    point_token = f"Point{well_code}_{fov_index:04d}"
    new_stem, n_subs = WELL_RE.subn(
        lambda m: f"{m.group(1)}{m.group(2)}_{point_token}",
        stem,
        count=1,
    )
    if n_subs == 0:
        return f"{stem}_{point_token}"
    return new_stem


def _resolve_bundle_axes(requested: str, available: dict) -> str:
    """Drop axes not present in the file (excluding y, x which are always required)."""
    requested = requested.lower().replace('v', '')
    resolved = ''.join(ax for ax in requested if ax in available or ax in ('y', 'x'))
    if 'y' not in resolved:
        resolved += 'y'
    if 'x' not in resolved:
        resolved += 'x'
    return resolved


def _validate_channel_order(
    channel_order: Sequence[int], axes: str, n_channels: int
) -> tuple:
    """Check that ``channel_order`` is a permutation of the available channels."""
    order = tuple(int(c) for c in channel_order)

    if 'c' not in axes:
        raise ValueError(
            f"channel_order was given but the resolved axes {axes!r} contain no "
            "channel axis"
        )
    if len(order) != n_channels:
        raise ValueError(
            f"channel_order has {len(order)} entries but the file has "
            f"{n_channels} channel(s)"
        )
    if sorted(order) != list(range(n_channels)):
        raise ValueError(
            f"channel_order must be a permutation of 0..{n_channels - 1}, "
            f"got {order}"
        )
    return order


def split_nd2(
    nd2_path: Union[str, Path],
    output_dir: Union[str, Path],
    bundle_axes: str = 'cyx',
    compression: str = 'zlib',
    overwrite: bool = False,
    channel_order: Union[Sequence[int], None] = None,
) -> list:
    """Split a multi-FOV .nd2 file into one .tif per field of view (point).

    For .nd2 files with a single field of view, a single .tif is written.
    The output filename always contains a ``Point<WellCode>_<FOVIndex>``
    token, regardless of whether the input was already split per point or
    not (see :func:`_build_output_stem`).

    Parameters
    ----------
    nd2_path : str or Path
        Path to the input .nd2 file.
    output_dir : str or Path
        Directory where the .tif files will be written. Created if missing.
    bundle_axes : str
        Axes to bundle together for each FOV. Default ``'cyx'`` for
        multi-channel 2D images. Use ``'yx'`` for grayscale 2D images,
        or ``'czyx'`` / ``'tczyx'`` for z-stacks / time series.
        The ``'v'`` axis is always iterated over and is ignored if passed.
    compression : str
        Compression for the output .tif files (e.g. ``'zlib'``, ``'lzw'``).
    overwrite : bool
        If False (default), skip writing FOVs whose output file already
        exists.
    channel_order : sequence of int, optional
        New channel order, as a permutation of the original channel indices.
        E.g. ``[4, 2, 0, 1, 3]`` writes the original channels 4, 2, 0, 1, 3
        as the new channels 0, 1, 2, 3, 4. Requires a channel axis in
        ``bundle_axes``.

    Returns
    -------
    list of Path
        Paths to the .tif files that were written (or that already existed
        when ``overwrite=False``).
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    well_match = WELL_RE.search(nd2_path.stem)
    well_code = well_match.group(2) if well_match else None

    written: list = []
    with ND2Reader(str(nd2_path)) as images:
        n_fov = int(images.sizes.get('v', 1))

        resolved_axes = _resolve_bundle_axes(bundle_axes, images.sizes)
        images.bundle_axes = resolved_axes

        channel_axis = None
        if channel_order is not None:
            channel_order = _validate_channel_order(
                channel_order, resolved_axes, int(images.sizes.get('c', 1))
            )
            channel_axis = resolved_axes.index('c')

        # NOTE: We deliberately do NOT use ``images.iter_axes = 'v'`` here.
        # In nd2reader, combining iter_axes with bundle_axes can leave the
        # first iterated frame as all zeros, which manifests as FOV 0 of
        # every file being a tiny ~1 KB tif when zlib-compressed. Pinning
        # the 'v' coordinate via ``default_coords`` and reading
        # ``images[0]`` for each FOV avoids that code path.
        has_v_axis = 'v' in images.sizes

        for v in tqdm(range(n_fov), desc=nd2_path.name, leave=False):
            out_stem = _build_output_stem(nd2_path.stem, v, well_code)
            out_path = output_dir / f"{out_stem}.tif"

            if out_path.exists() and not overwrite:
                written.append(out_path)
                continue

            if has_v_axis:
                images.default_coords['v'] = v
            frame = np.asarray(images[0])

            if channel_order is not None:
                frame = np.take(frame, channel_order, axis=channel_axis)

            tifffile.imwrite(
                out_path,
                frame.astype(np.uint16),
                compression=compression,
                imagej=True,
                metadata={'axes': resolved_axes.upper()},
            )
            written.append(out_path)

    return written


def split_nd2_folder(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    bundle_axes: str = 'cyx',
    compression: str = 'zlib',
    overwrite: bool = False,
    channel_order: Union[Sequence[int], None] = None,
) -> None:
    """Split every .nd2 file in ``input_dir`` into per-FOV .tif files.

    Errors on individual files are caught and printed so that processing
    continues with the remaining files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == '.nd2' and not f.stem.startswith('.')
    )

    if not files:
        print(f"No .nd2 files found in {input_dir}")
        return

    for f in tqdm(files, desc=f"Splitting nd2 files in {input_dir.name}"):
        try:
            split_nd2(
                f,
                output_dir,
                bundle_axes=bundle_axes,
                compression=compression,
                overwrite=overwrite,
                channel_order=channel_order,
            )
        except Exception as e:
            print(f"Error splitting {f.name}: {e}")
