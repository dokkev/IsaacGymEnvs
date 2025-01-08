import torch
import gym
from gym import Wrapper

class RolloutLogger(Wrapper):
    """
    A Gym wrapper that logs transitions (s, a, r, s') at each step.
    """
    def __init__(self, env, save_to_disk=True, save_path='rollouts.pt'):
        super().__init__(env)
        self.rollouts = []  # List to store transitions
        self.last_obs = None

        self.save_to_disk = save_to_disk
        self.save_path = save_path

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
   

        # Get the number of environments
        num_envs = obs['obs'].shape[0]
        
        # print(done)
        

        for i in range(num_envs):
            curr_state_i = self.last_obs['obs'][i]
            priv_info_i = self.last_obs['priv_info'][i]
            prop_history_i = self.last_obs['proprio_hist'][i]
            # print dimensions of prop_history_i
            action_i = action[i]
            reward_i = reward[i]
            next_state_i = obs['obs'][i]
            done_i = done[i]
            env_id = i
               
            
    

            # Append the transition as a dictionary
            self.rollouts.append({
                'env_id': env_id,
                # 'state': curr_state_i,
                # 'action': action_i,
                # 'reward': reward_i,
                'proprio_hist': prop_history_i,
                'priv_info': priv_info_i,
                # 'next_state': next_state_i,
                'done': done_i
            })

        # Save to disk periodically
        if self.save_to_disk and len(self.rollouts) % 1000 == 0:
            self.save_rollouts()

        # Update last_obs for the next step
        self.last_obs = obs
        return obs, reward, done, info

    def save_rollouts(self):
        # Save rollouts as a .pt file
        torch.save(self.rollouts, self.save_path)
        print(f"[RolloutLogger] Saved rollouts to {self.save_path}.")

    def get_rollouts(self):
        return self.rollouts

    def clear_rollouts(self):
        self.rollouts = []
