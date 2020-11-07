from pathlib import Path

import yaml

UV_LOOKUP_TABLE = Path("assets") / "dp_uv_lookup_256.npy"

with open(Path(__file__).parent / "params.yaml") as config_file:
    CONFIG = yaml.safe_load(config_file)
