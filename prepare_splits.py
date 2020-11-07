import random
from pathlib import Path

import click
import numpy as np
from functional import seq

np.random.seed(0)
random.seed(0)


def _extract_parent(path: Path) -> str:
    parent = path.parent
    while "id_" not in parent.name:
        parent = parent.parent
    return parent.name


@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
@click.option(
    "--ratio",
    type=float,
    help=(
        "Ratio of samples to be assigned as training samples. The rest will "
        "be validation"
    ),
    default=0.8,
)
def prepare_splits(input_dir: str, output_dir: str, ratio: int):
    output_path = Path(output_dir)
    files = list(Path(input_dir).rglob("*.h5*"))
    print("{} files to split".format(len(files)))

    groups = np.array(
        seq(files).group_by(_extract_parent).list(), dtype="object"
    )

    indices = np.random.permutation(len(groups))

    train_indices = indices[: int(ratio * len(indices))]
    valid_indices = indices[int(ratio * len(indices)) :]

    (output_path / "train.txt").write_text(
        "\n".join(
            seq(groups[train_indices])
            .flat_map(lambda x: x[1])
            .map(Path.as_posix)
        )
    )
    (output_path / "valid.txt").write_text(
        "\n".join(
            seq(groups[valid_indices])
            .flat_map(lambda x: x[1])
            .map(Path.as_posix)
        )
    )


if __name__ == "__main__":
    prepare_splits()
