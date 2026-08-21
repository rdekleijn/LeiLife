"""Recording and animation of a single verbose agent run.

Unlike the original module-level globals + import-time `plt.figure()`, all
state here is scoped to a `RunRecorder` instance and the matplotlib figure
is only created when `animate_run` is actually called. That makes it safe
to import this module inside joblib worker processes that never animate
anything.
"""

import csv


class RunRecorder:
    def __init__(self):
        self.agent_xpos = []
        self.agent_ypos = []
        self.food_xpos = []
        self.food_ypos = []
        self.orientation = []
        self.token_count = []

    def record(self, env):
        agent = env.agents[0]
        food = env.foodtokens[0]
        self.agent_xpos.append(agent.location[0])
        self.agent_ypos.append(agent.location[1])
        self.food_xpos.append(food.location[0])
        self.food_ypos.append(food.location[1])
        self.orientation.append(agent.orientation)
        self.token_count.append(agent.eatenTokens)

    def __len__(self):
        return len(self.agent_xpos)

    def to_csv(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['agent_x', 'agent_y', 'food_x', 'food_y', 'orientation'])
            for i in range(len(self)):
                writer.writerow([
                    self.agent_xpos[i], self.agent_ypos[i],
                    self.food_xpos[i], self.food_ypos[i],
                    self.orientation[i],
                ])


def animate_run(recorder, env_size, out_path, fps=30):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                          xlim=(0, env_size), ylim=(0, env_size))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    food, = ax.plot([], [], '*r', lw=3)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, food, time_text

    def step(i):
        line.set_data([recorder.agent_xpos[i]], [recorder.agent_ypos[i]])
        food.set_data([recorder.food_xpos[i]], [recorder.food_ypos[i]])
        time_text.set_text('Tokens Eaten: %.0f' % recorder.token_count[i])
        return line, food, time_text

    ani = animation.FuncAnimation(
        fig, step, frames=len(recorder), interval=1000. / fps,
        blit=False, init_func=init,
    )
    ani.save(out_path, fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)


class GroupRunRecorder:
    """Like RunRecorder, but for a shared environment with multiple agents.

    Also records each agent's emitted_signal per tick (stays 0 for
    non-signaling agents) alongside food and agent positions, which is
    what analysis/signal_analysis.py needs to relate signal content to
    world state (e.g. "does this agent's signal track food direction?",
    computed from the recorded positions) and to compare signal usage
    across different groups' saved replays.
    """

    def __init__(self):
        self.agent_xpos = []  # one entry per tick, each a list over agents
        self.agent_ypos = []
        self.agent_orientation = []
        self.food_xpos = []  # one entry per tick, each a list over food tokens
        self.food_ypos = []
        self.token_counts = []  # one entry per tick, each a list over agents
        self.signals = []  # one entry per tick, each a list over agents
        self.agent_ids = []  # one entry per tick, each a list over agents

    def record(self, env):
        self.agent_xpos.append([a.location[0] for a in env.agents])
        self.agent_ypos.append([a.location[1] for a in env.agents])
        self.agent_orientation.append([a.orientation for a in env.agents])
        self.food_xpos.append([f.location[0] for f in env.foodtokens])
        self.food_ypos.append([f.location[1] for f in env.foodtokens])
        self.token_counts.append([a.eatenTokens for a in env.agents])
        self.signals.append([a.emitted_signal for a in env.agents])
        self.agent_ids.append([a.agent_id for a in env.agents])

    def __len__(self):
        return len(self.agent_xpos)

    def to_csv(self, path):
        # Raw food positions don't line up row-for-row with the
        # one-row-per-agent shape below (there can be a different number
        # of food tokens than agents), so instead of exporting them
        # separately, each agent's row gets its distance/bearing to the
        # *nearest* food token -- directly comparable to what
        # update_visual_field senses, and what analysis/signal_analysis.py
        # correlates emitted signal against.
        from .geometry import calc_dir_and_dist

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tick', 'agent_idx', 'agent_id', 'agent_x', 'agent_y', 'orientation',
                              'eaten_tokens', 'signal', 'nearest_food_dist', 'nearest_food_bearing'])
            for tick in range(len(self)):
                fx, fy = self.food_xpos[tick], self.food_ypos[tick]
                for i, (agent_id, x, y, orientation, tokens, signal) in enumerate(zip(
                        self.agent_ids[tick], self.agent_xpos[tick], self.agent_ypos[tick],
                        self.agent_orientation[tick], self.token_counts[tick], self.signals[tick])):
                    if fx:
                        bearing, dist = min(
                            (calc_dir_and_dist(fx[j] - x, fy[j] - y, orientation) for j in range(len(fx))),
                            key=lambda bearing_dist: bearing_dist[1],
                        )
                    else:
                        bearing, dist = None, None
                    writer.writerow([tick, i, agent_id, x, y, orientation, tokens, signal, dist, bearing])


def animate_group_run(recorder, env_size, out_path, fps=30):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                          xlim=(0, env_size), ylim=(0, env_size))
    ax.grid()
    agent_scatter = ax.scatter([], [], c='tab:blue', s=60, zorder=2)
    food_scatter = ax.scatter([], [], c='red', marker='*', s=120, zorder=1)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        agent_scatter.set_offsets(np.empty((0, 2)))
        food_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return agent_scatter, food_scatter, time_text

    def step(i):
        agent_scatter.set_offsets(np.column_stack([recorder.agent_xpos[i], recorder.agent_ypos[i]]))
        food_scatter.set_offsets(np.column_stack([recorder.food_xpos[i], recorder.food_ypos[i]]))
        time_text.set_text('Total Tokens Eaten: %.0f' % sum(recorder.token_counts[i]))
        return agent_scatter, food_scatter, time_text

    ani = animation.FuncAnimation(
        fig, step, frames=len(recorder), interval=1000. / fps,
        blit=False, init_func=init,
    )
    ani.save(out_path, fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)
