# main idea for building mountain car environment is from gym library mountain_car.py file
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
# another source for building environmennt:
# https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/environments/mountain_car.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MountainCar:
    def __init__(self):
        # set position range
        self.position_max = 0.5
        self.position_min = -1.2
        # set velocy range
        self.velocity_max = 0.07
        self.velocity_min = -0.07
        # set engine force and the gravity
        self.engine_force = 0.001
        self.gravity = 0.0025
        # 
        self.target_position = 0.5
        self.target_velocity = 0.0
        
    
    def reset(self):
        position = np.random.uniform(-0.6, -0.4)
        velocity = 0
        self.state = np.array([position, velocity])
        return self.state
    
    def step(self, action):
        position, velocity = self.state
        ## not sure using action or (action - 1)
        velocity += action * self.engine_force - self.gravity * np.cos(3 * position)
        velocity = np.clip(velocity, self.velocity_min, self.velocity_max)
        position += velocity
        position = np.clip(position, self.position_min, self.position_max)
        
        if (position == self.position_min and velocity < 0):
            velocity = 0

        if (position >= self.target_position and velocity >= self.target_velocity):
            done = True
            
        reward = -1.0

        self.state = [position, velocity]
        return np.array(self.state), reward, done
    

        