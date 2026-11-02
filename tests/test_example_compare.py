import subprocess
import sys
from pathlib import Path

from alife.experiment import run_group_evolution


def _run_tiny_experiment(output_dir, kin_based_groups, seed=1):
    run_group_evolution(
        num_agents=8, num_gens=1, lifetime=605, env_size=40, group_size=4,
        n_hidden=4, top_n=2, offspring_per=4, mutation_std=0.3,
        signaling=True, social_learning=True, seed=seed, num_cores=1,
        kin_based_groups=kin_based_groups, save_best_every=999,
        save_all_groups_every=1, output_dir=str(output_dir),
    )


def test_compare_kin_vs_mixed_script_runs_end_to_end(tmp_path):
    kin_dir = tmp_path / "kin"
    mixed_dir = tmp_path / "mixed"
    _run_tiny_experiment(kin_dir, kin_based_groups=True)
    _run_tiny_experiment(mixed_dir, kin_based_groups=False)

    script = Path(__file__).resolve().parent.parent / "examples" / "compare_kin_vs_mixed.py"
    plot_path = tmp_path / "fitness.png"
    result = subprocess.run(
        [sys.executable, str(script), str(kin_dir), str(mixed_dir), "--fitness-plot", str(plot_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Kin-based grouping" in result.stdout
    assert "Mixed grouping" in result.stdout
    assert plot_path.exists()


def test_compare_kin_vs_mixed_handles_missing_group_run_replays(tmp_path, capsys):
    # save_all_groups_every not set -> no per-group replay CSVs -> the
    # dialect report should degrade gracefully, not crash.
    kin_dir = tmp_path / "kin_no_replays"
    run_group_evolution(
        num_agents=8, num_gens=1, lifetime=605, env_size=40, group_size=4,
        n_hidden=4, top_n=2, offspring_per=4, signaling=True, social_learning=True,
        seed=1, num_cores=1, kin_based_groups=True, save_best_every=999,
        output_dir=str(kin_dir),
    )
    mixed_dir = tmp_path / "mixed_no_replays"
    run_group_evolution(
        num_agents=8, num_gens=1, lifetime=605, env_size=40, group_size=4,
        n_hidden=4, top_n=2, offspring_per=4, signaling=True, social_learning=True,
        seed=1, num_cores=1, kin_based_groups=False, save_best_every=999,
        output_dir=str(mixed_dir),
    )

    script = Path(__file__).resolve().parent.parent / "examples" / "compare_kin_vs_mixed.py"
    result = subprocess.run(
        [sys.executable, str(script), str(kin_dir), str(mixed_dir)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "no save_all_groups_every replays found" in result.stdout
