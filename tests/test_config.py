import random

import numpy as np

from hd.config import load_config, resolve_path, set_seed


def test_config_loads_and_splits_sum_to_one():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert abs(sum(cfg["splits"][k] for k in ("train", "val", "test")) - 1.0) < 1e-9


def test_resolve_path_is_inside_project():
    cfg = load_config()
    assert resolve_path(cfg, "splits").name == "splits"


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    b = (random.random(), np.random.rand())
    assert a == b
