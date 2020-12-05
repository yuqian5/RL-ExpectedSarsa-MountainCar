# main idea for building mountain car environment is from gym library mountain_car.py file
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
# another source for building environmennt:
# https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/environments/mountain_car.py
# one more source:
# https://github.com/ChanchalKumarMaji/Reinforcement-Learning-Specialization/blob/master/Prediction%20and%20Control%20with%20Function%20Approximation/Week%203/Notebook:%20Function%20Approximation%20and%20Control/mountaincar_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class MountainCar:
    def __init__(self):
        # seed rnd gen
        np.random.seed()
        # set position range
        self.position_max = 0.5
        self.position_min = -1.2
        # set velocy range
        self.velocity_max = 0.07
        self.velocity_min = -0.07
        # set engine force and the gravity
        self.engine_force = 0.001
        self.gravity = 0.0025
        # set target
        self.target_position = 0.5
        self.target_velocity = 0.0
        # create position list for render build
        self.position_list = []
        # the time step in seconds (default 0.1)
        self.delta_t = 0.01
        # set the draft state
        # self.state = np.array([np.random.uniform(-0.6, -0.4), 0.0])

    def env_init(self, agent_info={}):
        local_obesrvation = 0.0
        self.reward_obs_term = (0.0, local_obesrvation, False)

    def reset(self):
        '''
        generate and reset position and velocity to start
        position: Each episode started from a random position [-0.6,-0.4)
        velocity: Each episode started from zero
        '''
        position = np.random.uniform(-0.6, -0.4)
        velocity = 0.0
        self.position_list.clear()
        self.state = np.array([position, velocity])

        # reseed rnd gen
        # np.random.seed()

        return self.state

    def step(self, action):
        '''
        One step in the envrionment by action.
        action: There are three actions [-1, 0, 1],
          -1 = move back
          0 = not move
          1 = move forward
        
        return: state, reward, done
        state: state = array([position, velocity])
              position: -1.2 <= position <= 0.5 
              velocity: -0.7 <= velocity<= 0.7
        reward: reward is always -1 per step
        done: a boolean to check if terminate
        '''
        done = False
        reward = -1.0
        position, velocity = self.state
        velocity += (action - 1) * self.engine_force - self.gravity * np.cos(3 * position)
        velocity = np.clip(velocity, self.velocity_min, self.velocity_max)
        position += velocity
        position = np.clip(position, self.position_min, self.position_max)
        self.position_list.append(position)

        if (position == self.position_min):
            velocity = 0.0
        elif (position == self.target_position):
            done = True
            reward = 0.0

        self.state = np.array([position, velocity])
        return self.state, reward, done

    # get the hight of the position
    # the fuction for the hight is get from issecting-reinforcement-learning:
    # https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/environments/mountain_car.py
    def p_height(self, position):
        '''
        This is the fuction which used to calculate hight of the position.
        '''
        return np.sin(3 * position) * 0.45 + 0.55

    # this part of code is modified the code from dissecting-reinforcement-learning:
    # https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/environments/mountain_car.py
    def render(self, file_path='./mountain_car.gif', mode='gif'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(-1.2, 0.5), ylim=(0, 1.2))
        ax.grid(False)  # disable the grid
        xa = np.linspace(self.position_max, self.position_min, num=100)
        ya = self.p_height(xa)

        ax.plot(xa, ya)

        # set the car
        dot, = ax.plot([], [], 'ro')
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
            x = self.position_list[i]
            y = self.p_height(x)
            dot.set_data(x, y)
            time_text.set_text("Time: " + str(np.round(i * self.delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return dot, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.position_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1 / self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1 / self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)
