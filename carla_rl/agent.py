import gym
import numpy as np
import torch
import torch.nn as nn

from .carla_env import CarlaEnv
from .env_wrappers import PreprocessCARLAObs


class MixedDQNAgent(nn.Module):
    div_epsilon = 1e-2


    def __init__(self, env, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.env_observation_space = env.observation_space
        self.state_shape = env.observation_space.shape
        assert len(self.state_shape) == 1
        self.state_shape = self.state_shape[0]
        self.n_actions_throttle = env.action_space[0].n
        self.steer_action_high = env.action_space[1].high
        self.steer_action_low = env.action_space[1].low

        layer_sizes =  [self.state_shape, 25, 15, 10]
        self.shared_model = nn.Sequential(*self._construct_layers(layer_sizes))
        self.mean_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], self.n_actions_throttle),
            nn.Tanh()
        )
        self.std_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], self.n_actions_throttle),
            nn.Sigmoid()
        )
        self.scale_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], self.n_actions_throttle),
            nn.ReLU()
        )
        self.add_head = nn.Linear(layer_sizes[-1], self.n_actions_throttle)


    def _construct_layers(self, layer_sizes):
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(self.state_shape, layer_size))
            else:
                layers.append(nn.Linear(layer_sizes[i - 1], layer_size))

            if i < len(layer_sizes) - 1:
                if i % 2 == 1:
                    layers.append(nn.BatchNorm1d(layer_size))
                layers.append(nn.ReLU())
        return layers


    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of lane lines, shape = [batch_size, lane_lines_data_count]
        """
        shared_params = self.shared_model(state_t)
        means = self.mean_head(shared_params)
        stds = self.std_head(shared_params) + MixedDQNAgent.div_epsilon
        # if (stds < 0).any():
        #     print(stds)
        #     print("HOW THE FUCK DID THIS HAPPEN?", state_t)
        #     assert False
        scalers = self.scale_head(shared_params)
        add = self.add_head(shared_params)
        res = torch.stack([means, stds, scalers, add], dim=2)
        # print("CHEEEEEEEEEEEEEEEEEEEEEEEEEEEECK")
        # print(res)
        # print(means)
        # print(stds)

        assert res.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(res.shape) == 3 and res.shape[1] == self.n_actions_throttle and res.shape[2] == 4

        return res


    def get_qvalue_params(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalue_params = self.forward(states)
        return qvalue_params.data.cpu().numpy()


    def sample_actions(self, qvalue_params):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_throttle_actions, _ = qvalue_params.shape

        random_throttle_actions = np.random.choice(n_throttle_actions, size=batch_size)
        random_steer_actions = np.random.uniform(low=self.steer_action_low, high=self.steer_action_high,
                                                    size=batch_size)
        random_actions = np.stack([random_throttle_actions, random_steer_actions], axis=1)
        max_score = 1 / (np.sqrt(2 * torch.pi) * qvalue_params[..., 1]) * qvalue_params[..., 2] + qvalue_params[..., 3]
        best_throttle_actions = max_score.argmax(axis=1)
        best_steer_actions = qvalue_params[np.arange(qvalue_params.shape[0]), best_throttle_actions, 0]
        best_actions = np.stack([best_throttle_actions, best_steer_actions], axis=1)

        should_explore = np.random.choice([0, 1], batch_size, p=[1-epsilon, epsilon])
        should_explore = np.stack([should_explore, should_explore], axis=1)
        return np.where(should_explore, random_actions, best_actions)
    

    def get_qvalues_from_params(self, params, actions):
        batch_size = params.shape[0]
        throttle = actions[:, 0].type(torch.long)   # shape: [batch_size]
        steer = actions[:, 1]                       # shape: [batch_size]

        # print("HMMMMMMM?", batch_size, throttle)
        mean = params[np.arange(batch_size), throttle, 0]           # shape: [batch_size]
        std = params[np.arange(batch_size), throttle, 1]            # shape: [batch_size]
        scale = params[np.arange(batch_size), throttle, 2]          # shape: [batch_size]
        add = params[np.arange(batch_size), throttle, 3]            # shape: [batch_size]

        exp_value = torch.exp(-0.5 * ((steer - mean) / std) ** 2)
        gaussian_values = 1 / (np.sqrt(torch.pi * 2) * std) * exp_value
        # print("RESULT OF HM", gaussian_values)
        # print("EXP", exp_value)
        # print("MEAN", mean)
        # print("STD", std)
        # if (std < 0).any():
        #     print("HOW????????????????????????")
        #     print(params[..., 1])
        #     print(params[np.arange(batch_size), throttle, 1])
        #     assert False
        # print("SCALE", scale)
        # print("ADD", add)
        # print("=======================================")
        return gaussian_values * scale + add


if __name__ == "__main__":
    # Sanity checks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = CarlaEnv(town="Town02", fps=20, im_width=1280, im_height=720, repeat_action=1, start_transform_type="random",
                   sensors="rgb", action_type="mixed", enable_preview=False, steps_per_episode=500, playing=False,
                   timeout=60)
    env = PreprocessCARLAObs(env)

    agent = MixedDQNAgent(env=env, epsilon=0.5)
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    print("STATE?", state)
    qvalues = agent.get_qvalue_params(np.concatenate([state, state]))
    print("QVALUES", qvalues)
    print("SAMPLED ACTIONS", agent.sample_actions(qvalues))