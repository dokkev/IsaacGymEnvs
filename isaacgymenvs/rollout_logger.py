import torch
import gym
from gym import Wrapper

class RolloutLogger(Wrapper):
    """
    A Gym wrapper that logs transitions:
    - Only when the cube is in contact with the end effector
    - Only for the first N timesteps of each trial
    - Trials are separated by done signals
    """
    def __init__(self, env, save_to_disk=True, save_path='rollouts.pt', max_timesteps_per_trial=10):
        super().__init__(env)
        self.rollouts = []  # List to store transitions
        self.last_obs = None
        self.save_to_disk = save_to_disk
        self.save_path = save_path
        self.cube_size = 0.05  # From the environment definition
        self.max_timesteps_per_trial = max_timesteps_per_trial
        
        # Create a dictionary to track timesteps for each environment
        self.trial_timesteps = {}  # env_id -> timesteps in current trial

    def is_in_contact(self, obs):
        """Check if the cube is in contact with the end effector using current observations.
        obs indices: cube_pos (0:3), eef_pos (7:10)"""
        cube_pos = obs[0:3]
        eef_pos = obs[7:10]
        distance = torch.norm(cube_pos - eef_pos)
        return distance < (self.cube_size * 10)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        # Reset trial timesteps for all environments
        num_envs = obs['obs'].shape[0]
        self.trial_timesteps = {i: 0 for i in range(num_envs)}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get the number of environments
        num_envs = obs['obs'].shape[0]

        for i in range(num_envs):
            curr_state_i = self.last_obs['obs'][i]
            priv_info_i = self.last_obs['priv_info'][i]
            prop_history_i = self.last_obs['proprio_hist'][i]
            done_i = done[i]
            env_id = i
            
            # Reset timestep counter if this trial is done
            if done_i:
                self.trial_timesteps[env_id] = 0
            else:
                self.trial_timesteps[env_id] += 1
                
                # Break if we hit max timesteps
                # if self.trial_timesteps[env_id] == self.max_timesteps_per_trial:
                #     breakpoint()
            
            # Only log if:
            # 1. Cube is in contact with end effector
            # 2. Within the max timesteps for this trial
            if (self.is_in_contact(curr_state_i) and 
                self.trial_timesteps[env_id] <= self.max_timesteps_per_trial):
                print("contat phase")
                
                self.rollouts.append({
                    'env_id': env_id,
                    'proprio_hist': prop_history_i,
                    'priv_info': priv_info_i,
                    'done': done_i,
                    'trial_timestep': self.trial_timesteps[env_id]
                })

        # Save to disk periodically
        if self.save_to_disk and len(self.rollouts) % 20 == 0:
            self.save_rollouts()

        # Update last_obs for the next step
        self.last_obs = obs
        return obs, reward, done, info

    def save_rollouts(self):
        # Save rollouts as a .pt file
        torch.save(self.rollouts, self.save_path)
        print(f"[RolloutLogger] Saved {len(self.rollouts)} contact transitions to {self.save_path}.")

    def get_rollouts(self):
        return self.rollouts

    def clear_rollouts(self):
        self.rollouts = []