import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

env_size = 100

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, env_size), ylim=(0, env_size))
ax.grid()
x = np.arange(0, env_size, 0.1)
line, = ax.plot([], [], 'o-', lw=2)
food, = ax.plot([], [], '*r', lw=3)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init_animation():
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i, df):
    #global agent_xpos, agent_ypos, food_xpos, food_ypox, tokenCount
    line.set_data(df['agent_xpos'][i], df['agent_ypos'][i])
    #print agent_xpos[i], agent_ypos[i]
    # sometimes there's no food...just put it offscreen?
    food.set_data(df['food_xpos'][i], df['food_ypos'][i])
    time_text.set_text('Tokens Eaten: %.0f' % df['tokenCount'][i]) #
    #energy_text.set_text('food_y = %.1f' % food_ypos[i]) # orientation? dist traveled?
    return line, time_text, energy_text

def save_movie(fname, df, lifetime, env_size):
    dt = 1./60
    interval = 1000 * dt #- (t1 - t0)
    ani = animation.FuncAnimation(fig, animate, frames=lifetime, fargs=(df,),
        interval=interval, blit=False, init_func=init_animation)
    # To save as an mp4 requires ffmpeg or mencoder to be installed.
    # The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this:
    # more info: http://matplotlib.sourceforge.net/api/animation_api.html
    ani.save('output/'+fname+'_agent_run.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    #plt.show()
