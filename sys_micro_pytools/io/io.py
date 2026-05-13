from pathlib import Path
import shutil
import subprocess
from nd2reader import ND2Reader
import tifffile
from bioio import BioImage
import bioio_lif
from tqdm import tqdm


def read_nd2(filename, bundle_axes='cyx'):
    with ND2Reader(filename) as images:
        if bundle_axes is not None:
            images.bundle_axes = bundle_axes
            images = images[0]
        return images
    

def read_tiff_or_nd2(filename, bundle_axes: str = 'cyx'):
    filename = Path(filename)
    if filename.suffix == '.tif':
        img = tifffile.imread(str(filename))
    elif filename.suffix == '.nd2':
        img = read_nd2(str(filename), bundle_axes)
    return img


def split_lif(
    lif_path: str | Path, 
    output_dir: str | Path, 
    bundle_axes: str = 'TZCYX',
    compression: str = 'zlib',
    do_create_subdir: bool = True,
    ) -> None:
    ''' Split a .lif file with multiple images into individual images stored as compressed .tif files.

    Parameters
    ----------
    lif_path : str or Path
        Path to the .lif file.
    output_dir : str or Path
        Directory where the individual .tif files will be stored.
    compression : str
        Compression algorithm to use for the .tif files.
    do_create_subdir : bool
        Whether to create a subdirectory in output_dir with the name of the lif file.
    '''
    lif_path = Path(lif_path)
    output_dir = Path(output_dir)
    
    if do_create_subdir:
        output_dir = output_dir / lif_path.stem
        
    output_dir.mkdir(parents=True, exist_ok=True)

    img = None
    try:
        img = BioImage(lif_path, reader=bioio_lif.Reader)
    except Exception as e:
        raise RuntimeError(f"Failed to read LIF file: {e}") from e

    for scene in tqdm(img.scenes, desc=f"Splitting {lif_path.name}"):
        img.set_scene(scene)
        data = img.get_image_data(dimension_order_out=bundle_axes)
        
        # Clean up scene name for filename
        scene_name = scene.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')
        
        if do_create_subdir:
            output_filename = f"{scene_name}.tif"
        else:
            output_filename = f"{lif_path.stem}_{scene_name}.tif"
            
        output_path = output_dir / output_filename
        
        # Write the image data to a compressed .tif file
        tifffile.imwrite(output_path, data, compression=compression, imagej=True)


def _discover_bioformats_series(
    lif_path: Path,
    showinf_bin: str = "showinf",
) -> list[int]:
    if shutil.which(showinf_bin) is None:
        raise RuntimeError(
            f"'{showinf_bin}' was not found on PATH. Install Bio-Formats tools, "
            "or pass --series explicitly."
        )

    cmd = [showinf_bin, "-nopix", str(lif_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to list series via showinf: {stderr}")

    series: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # Typical line starts with "Series #0"
        if line.startswith("Series #"):
            token = line.split("Series #", 1)[1].split()[0]
            token = token.rstrip(":")
            if token.isdigit():
                series.append(int(token))

    if not series:
        raise RuntimeError(
            "Could not detect series in showinf output. "
            "Please pass --series explicitly."
        )

    return sorted(set(series))


def split_lif_bioformats(
    lif_path: str | Path,
    output_dir: str | Path,
    compression: str = "LZW",
    do_create_subdir: bool = True,
    series: list[int] | None = None,
    bfconvert_bin: str = "bfconvert",
    showinf_bin: str = "showinf",
) -> None:
    """
    Split .lif file into per-series OME-TIFF files using Bio-Formats tools.

    This path is useful for Leica datasets where readlif/bioio_lif do not expose
    dimensions (for example mirror/time planes) the same way as Fiji.
    """
    lif_path = Path(lif_path)
    output_dir = Path(output_dir)

    if shutil.which(bfconvert_bin) is None:
        raise RuntimeError(
            f"'{bfconvert_bin}' was not found on PATH. Install Bio-Formats tools first."
        )

    if do_create_subdir:
        output_dir = output_dir / lif_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if series is None:
        series = _discover_bioformats_series(lif_path, showinf_bin=showinf_bin)

    for s in tqdm(series, desc=f"Bio-Formats split {lif_path.name}"):
        out_name = f"{lif_path.stem}_series_{s:03d}.ome.tif"
        out_path = output_dir / out_name
        cmd = [
            bfconvert_bin,
            "-overwrite",
            "-series",
            str(s),
            "-compression",
            compression,
            str(lif_path),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"bfconvert failed for series {s}.\nCommand: {' '.join(cmd)}\n{stderr}"
            )
