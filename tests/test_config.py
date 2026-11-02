import pytest

from alife.config import ExperimentConfig


def test_defaults():
    config = ExperimentConfig()
    assert config.name == "experiment"
    assert config.num_agents == 100
    assert config.signaling is False
    assert config.save_mp4 is True


def test_from_yaml_round_trip(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("name: test_run\nnum_agents: 10\nsignaling: true\n")

    config = ExperimentConfig.from_yaml(str(yaml_path))

    assert config.name == "test_run"
    assert config.num_agents == 10
    assert config.signaling is True
    assert config.num_gens == 250  # unset fields keep their default


def test_from_yaml_rejects_unknown_keys(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("bogus_field: 1\n")

    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(str(yaml_path))


def test_save_provenance_writes_config_and_git_commit(tmp_path):
    config = ExperimentConfig(name="prov_test", num_agents=7)
    out_dir = tmp_path / "out"

    config.save_provenance(str(out_dir))

    written = (out_dir / "config.yaml").read_text()
    assert "prov_test" in written
    assert "num_agents: 7" in written
    assert "git_commit" in written
