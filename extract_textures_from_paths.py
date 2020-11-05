from pathlib import Path

import click
import h5py
import tqdm
from PIL import Image


@click.command()
@click.argument("in_path")
@click.argument("out_path")
def extract(in_path: str, out_path: str):
    h5_files = list(Path(in_path).rglob("*.h5*"))
    output_path = Path(out_path)
    output_path.mkdir(parents=True, exist_ok=True)

    index = 0
    for file_path in tqdm.tqdm(h5_files):
        with h5py.File(file_path.as_posix(), mode="r") as h5_file:
            textures = h5_file["textures"][:]

        for texture in textures:
            img = Image.fromarray(texture)
            img.save(output_path / "{}.png".format(index))

            index += 1


if __name__ == "__main__":
    extract()
