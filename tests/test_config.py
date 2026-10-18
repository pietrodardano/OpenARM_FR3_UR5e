"""NFR4: one strict config file, loaded into typed dataclasses."""

from __future__ import annotations

import pytest
import yaml

from robotokens.config import DEFAULT_CONFIG, ConfigError, load_config


def test_full_config_loads(cfg):
    assert cfg.robot.side == "left"
    assert len(cfg.hardware.kp) == 7
    assert cfg.hardware.send_can_ids[0] == 0x01
    assert cfg.task.weigh.min_z == 3.0
    assert set(cfg.workcell.bins) == {"left", "right"}


def test_table_height_in_base_frame(cfg):
    # 0.725 m tabletop, shoulder at 1.330 m, model shoulder 0.697 m above the base: 0.092 m.
    assert cfg.table_z == pytest.approx(0.092, abs=1e-9)


def _write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_unknown_key_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    data["hardware"]["kp_typo"] = 1.0
    with pytest.raises(ConfigError, match="hardware.kp_typo"):
        load_config(_write(tmp_path, data))


def test_missing_key_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    del data["task"]["weigh"]["min_z"]
    with pytest.raises(ConfigError, match="task.weigh.min_z"):
        load_config(_write(tmp_path, data))


def test_bad_side_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    data["robot"]["side"] = "middle"
    with pytest.raises(ConfigError, match="side"):
        load_config(_write(tmp_path, data))
