import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "run_experiment.py"

TINY_CONFIG = dict(
    name="cli_test",
    num_agents=8, num_gens=1, lifetime=605, env_size=40, group_size=4,
    n_hidden=4, top_n=2, offspring_per=4, mutation_std=0.3,
    signaling=True, social_learning=True, seed=1, num_cores=1,
    save_best_every=999, checkpoint_every=1,
)


def _run_cli(tmp_path, extra_args=()):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(TINY_CONFIG))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(config_path), *extra_args],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path / "output" / "cli_test"


def test_run_experiment_cli_writes_expected_output(tmp_path):
    output_dir = _run_cli(tmp_path)

    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "generation_log.csv").exists()
    assert (output_dir / "checkpoint.pkl").exists()

    written_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    assert written_config["name"] == "cli_test"
    assert written_config["git_commit"]  # non-empty


def test_run_experiment_cli_resume_flag_continues_past_num_gens(tmp_path):
    output_dir = _run_cli(tmp_path)  # ran num_gens=1 -> generation 0 only
    checkpoint_path = output_dir / "checkpoint.pkl"

    # Same config but num_gens bumped up, resuming from the checkpoint
    # written by the first (1-generation) run.
    config_path = tmp_path / "config.yaml"
    config = dict(TINY_CONFIG, num_gens=2)
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(config_path), "--resume", str(checkpoint_path)],
        cwd=str(tmp_path), capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    import pandas as pd
    df = pd.read_csv(output_dir / "generation_log.csv")
    assert set(df["gen"]) == {0, 1}
