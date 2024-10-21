import isaacgym
import isaacgymenvs
import torch


num_envs = 32


envs = isaacgymenvs.make(
	seed=0, 
	task="FrankaCubeSlide", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(10000000):
	actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	# actions = torch.zeros((num_envs,) + envs.action_space.shape, device = 'cuda:0') 
	envs.step(actions)