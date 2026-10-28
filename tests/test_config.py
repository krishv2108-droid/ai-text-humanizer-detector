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


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    from hd.config import load_env

    env = tmp_path / ".env"
    env.write_text('# comment\nHD_TEST_A=one\nexport HD_TEST_B="two"\nHD_TEST_C=from_file\n\nnot a pair\n')
    monkeypatch.delenv("HD_TEST_A", raising=False)
    monkeypatch.delenv("HD_TEST_B", raising=False)
    monkeypatch.setenv("HD_TEST_C", "already_set")
    assert load_env(env) is True
    import os

    assert os.environ["HD_TEST_A"] == "one" and os.environ["HD_TEST_B"] == "two"
    assert os.environ["HD_TEST_C"] == "already_set"  # an existing variable wins
    monkeypatch.delenv("HD_TEST_A")
    monkeypatch.delenv("HD_TEST_B")
    assert load_env(tmp_path / "missing.env") is False
