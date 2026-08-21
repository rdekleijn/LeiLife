import glob
import os

from alife.experiment import run_group_evolution
from analysis.plots import load_generation_log, plot_fitness_over_generations, build_lineage_table
from analysis.signal_analysis import load_group_run, signal_food_correlation, find_group_runs, compare_group_dialects


def test_analysis_pipeline_end_to_end(tmp_path):
    output_dir = str(tmp_path / "run")
    run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, save_all_groups_every=1,
        n_hidden=4, top_n=2, offspring_per=2, signaling=True, output_dir=output_dir,
    )

    # generation_log.csv -> fitness plot + lineage table, without re-running anything
    gen_df = load_generation_log(os.path.join(output_dir, 'generation_log.csv'))
    assert len(gen_df) == 4
    plot_fitness_over_generations(gen_df, out_path=str(tmp_path / "fitness.png"))
    assert (tmp_path / "fitness.png").exists()
    lineage = build_lineage_table(gen_df)
    assert len(lineage) == 4

    # per-tick group replays -> signal analysis
    group_run_paths = find_group_runs(output_dir, gen=0)
    assert len(group_run_paths) == 2  # 4 agents / group_size 2
    df = load_group_run(group_run_paths[0])
    assert {'signal', 'nearest_food_dist', 'nearest_food_bearing'} <= set(df.columns)
    signal_food_correlation(df)  # should not raise

    dialects = compare_group_dialects(group_run_paths)
    assert len(dialects) == 2
