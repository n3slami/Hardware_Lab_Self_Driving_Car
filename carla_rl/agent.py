import gym
import numpy as np
import torch
import torch.nn as nn

from .carla_env import CarlaEnv
from .env_wrappers import PreprocessCARLAObs


class DQNAgent(nn.Module):
    def __init__(self, env, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.state_shape = env.observation_space.shape
        self.n_actions_throttle = env.action_space[0].n
        self.n_actions_steer = env.action_space[1].n

        # TODO
        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        #<YOUR CODE>
        self.layers = nn.Sequential(
            nn.Conv2d(self.state_shape[0], 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, h, w, 4 * c]
        """
        # TODO
        # Use your network to compute qvalues for given state
        model_input = torch.swapaxes(state_t, 1, 3)
        model_input = torch.swapaxes(model_input, 2, 3)
        qvalues = self.layers(model_input)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions).squeeze(-1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = CarlaEnv(town="Town02", fps=20, im_width=1280, im_height=720, repeat_action=1, start_transform_type="random",
                   sensors="rgb", action_type="discrete", enable_preview=False, steps_per_episode=500, playing=False,
                   timeout=60)
    env = PreprocessCARLAObs(env)
    print(env.observation_space, env.observation_space.shape)
    print(env.action_space, env.action_space[0].n, env.action_space[1].n)
    print(env.action_space.sample())
