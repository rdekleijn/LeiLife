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
