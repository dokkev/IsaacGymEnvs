import torch
from gym import Wrapper

class StatePredicter(Wrapper):
    """
    A Gym wrapper that predicts the next state using a pre-trained model
    and computes the difference with the actual next state.
    """
    def __init__(self, env, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(env)
        self.device = torch.device(device)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.model = self._load_model(model_path)
        self.last_obs = None

    def _load_model(self, model_path):
        """
        Load a pre-trained model for state prediction.
        """
        class StateActionPredictor(torch.nn.Module):
            def __init__(self, input_dim, output_dim=7):
                super(StateActionPredictor, self).__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, output_dim)
                )

            def forward(self, x):
                return self.model(x)

        input_dim = self.obs_dim + self.action_dim
        model = StateActionPredictor(input_dim=input_dim, output_dim=7)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"[INFO] Loaded model from {model_path}")
        return model

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        return self.last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Concatenate the last observation and the current action
        state_action = torch.tensor(
            [*self.last_obs, *action],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Predict the next state
        predicted_next_state = self.model(state_action).detach().cpu().numpy()

        # Compute the difference between predicted and actual next state
        actual_next_state = obs  # Assuming the observation is the actual next state
        diff = predicted_next_state - actual_next_state[:7]  # Compare relevant elements
        diff_norm = torch.norm(torch.tensor(diff), p=2).item()  # L2 norm of the difference

        print(f"[INFO] Predicted Next State: {predicted_next_state}")
        print(f"[INFO] Actual Next State: {actual_next_state[:7]}")
        print(f"[INFO] Difference (L2 norm): {diff_norm}")

        self.last_obs = obs
        return obs, reward, done, info
