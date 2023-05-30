import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
import random

class CraneEnv(gym.Env[np.ndarray,Union[int,np.ndarray]]):
    metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
    }

    def __init__(self,render_mode:Optional[str]=None):
        self.g = 9.8    # 9.8
        self.M = 2.5    # 1.0
        self.m = 5    # 1.5
        self.l = 1.0    # 2.0
        self.EXPECT = 1
        # self.force_mag = 1
        self.tau = 0.01  # seconds between state updates
        self.t = 0.0
        self.E = [self.EXPECT]

        self.FORCE = np.linspace(-1,2,4)
        self.action_space = spaces.Discrete(len(self.FORCE))

        self.render_mode = render_mode

        self.screen_width = 800
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.miu = 0.2

        self.steps_beyond_terminated = None
        
    def step(self,action):
        
        x,dx,theta,dtheta = self.state

        force = action
        
        ddx = (self.l*self.m*math.sin(theta)*dtheta**2 + force - dx*self.miu + self.g*self.m*math.cos(theta)*math.sin(theta))\
            /(-self.m*math.cos(theta)**2 + self.M + self.m)
           
        ddtheta = -(self.l*self.m*math.cos(theta)*math.sin(theta)*dtheta**2 + force*math.cos(theta) \
                    - dx*self.miu*math.cos(theta) + self.g*self.m*math.sin(theta) + self.M*self.g*math.sin(theta))\
            /(self.l*(-self.m*math.cos(theta)**2 + self.M + self.m))
           
        dx = dx + self.tau * ddx   
        x = x + self.tau * dx
        dtheta = dtheta + self.tau * ddtheta
        theta = theta + self.tau * dtheta
        self.t += self.tau
        
        while 1:
            if theta >= 2*math.pi:
                theta -= 2*math.pi
            else:
                break
            
        while 1:
            if theta <= -2*math.pi:
                theta += 2*math.pi
            else:
                break
        
        e = abs(self.EXPECT - x)
        self.E.append(e)
            
        self.state = (x, dx, theta, dtheta)

        terminated = False
        if self.t >= 20:
            terminated = True
            self.t = 0

        # reward = - e - abs(theta) - 0.1*abs(action)
        reward = - e - abs(theta)
        if self.render_mode == "human":
            self.render()
        # return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
        return np.array(self.state, dtype=np.float32), reward, terminated, {}
            
    def reset(self,*,seed:Optional[int] = None,options:Optional[dict] = None):
        super().reset(seed=seed)

        self.state = [0.0,0.0,0.0,0.0]
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        import pygame
        import math
        
        pygame.init()
        length = 800
        width = 400
        screen = pygame.display.set_mode((length,width))
        
        pygame.display.set_caption('CraneDisplay')
        
        Seashell = 255,255,255
        
        Cart_Length = 50
        Load_Radius = 25
        Pole_Length = 200
        
        X = self.state
        # Pole_theta = 0.25
        Pole_theta = X[2]
        # Cart_Position = [400,width/4]
        Cart_Position = [X[0]+400,width/4]
        Load_Position = [Cart_Position[0]+Pole_Length*math.sin(Pole_theta),Cart_Position[1]+Pole_Length*math.cos(Pole_theta)]
        
        RECT = (Cart_Position[0]-Cart_Length/2,Cart_Position[1]-Cart_Length/2,Cart_Length,Cart_Length)
        
        screen.fill(Seashell)
        
        # pygame.draw.rect(screen, color=(0,0,128),rect=(0,width/8,length/16,2*width/8))
        # pygame.draw.rect(screen, color=(0,0,128),rect=(length/16*15,width/8,length,2*width/8))
        pygame.draw.line(screen, color=(0,0,128), start_pos=(0,width/4),
                         end_pos=(length,width/4),width=10)
        pygame.draw.rect(screen,color=(128,128,128),rect=RECT)
        pygame.draw.aaline(screen,color=(50,50,50),
                         start_pos=Cart_Position,
                         end_pos=Load_Position,
                         blend=100)
        pygame.draw.circle(screen, color=(128,128,128), center=Load_Position, radius=Load_Radius)
        pygame.draw.circle(screen, color=(255,0,0), center=(400,100), radius=2)
        pygame.draw.circle(screen, color=(255,0,0), center=(410,100), radius=2)
        pygame.display.update()
        
            
    def close(self):
        import pygame

        pygame.display.quit()
        pygame.quit()
        self.isopen = False        


