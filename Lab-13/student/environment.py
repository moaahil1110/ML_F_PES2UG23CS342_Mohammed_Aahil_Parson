from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2
from collections import deque


class MarioEnvironment:
    def __init__(self, env_name='SuperMarioBros-v0', frame_skip=4, stack_frames=4, render_mode='human'):
        self.env = gym_super_mario_bros.make(env_name)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.frame_skip = frame_skip
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        self.n_actions = self.env.action_space.n
        self.state_shape = (stack_frames, 84, 84)
    
    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        return frame
    
    def reset(self):
        initial_frame = self.env.reset()
        processed_frame = self.preprocess_frame(initial_frame)
        self.frame_stack.clear()
        for _ in range(self.stack_frames):
            self.frame_stack.append(processed_frame)
        return np.array(self.frame_stack)
    
    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.frame_skip):
            next_frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        processed_frame = self.preprocess_frame(next_frame)
        self.frame_stack.append(processed_frame)
        return np.array(self.frame_stack), total_reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        self.env.close()

