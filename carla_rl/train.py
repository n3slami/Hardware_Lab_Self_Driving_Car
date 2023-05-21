import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import time
import argparse

from .utils import *
from .carla_env import CarlaEnv
from .env_wrappers import PreprocessCARLAObs
from .agent import MixedDQNAgent
from .replay_buffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class MixedTrainer:
    LEARNING_RATE = 1e-5
    REPLAY_BUFFER_SIZE = 7000

    def __init__(self, env, agent, target_network):
        self.env = env
        self.agent = agent
        self.target_network = target_network
        self.exp_replay = ReplayBuffer(MixedTrainer.REPLAY_BUFFER_SIZE)
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=MixedTrainer.LEARNING_RATE)
    

    def compute_td_loss(self, states, actions, rewards, next_states, is_done,
                    gamma=0.95, check_shapes=False, device=device):
        """ Compute td loss using torch operations only. Use the formulae above. """
        states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

        # for some torch reason should not make actions a tensor
        actions = torch.tensor(actions, device=device, dtype=torch.float)   # shape: [batch_size, *action_shape]
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)   # shape: [batch_size, *action_shape]
        # shape: [batch_size, *state_shape]
        next_states = torch.tensor(next_states, device=device, dtype=torch.float)
        is_done = torch.tensor(
            is_done.astype('float32'),
            device=device,
            dtype=torch.float
        )  # shape: [batch_size]
        is_not_done = 1 - is_done

        # print("AGENT SHIT")
        # get q-value params for all actions in current states
        predicted_qvalue_params = self.agent(states)
        # get q-values for chosen actions
        predicted_qvalues_for_actions = self.agent.get_qvalues_from_params(predicted_qvalue_params, actions)

        # print("TARGET NETWORK SHIT")
        # compute q-values for all actions in next states
        predicted_next_qvalue_params = self.target_network(next_states)
        # compute V*(next_states) using predicted next q-value params
        next_actions = self.target_network.sample_actions(predicted_next_qvalue_params.data.cpu().numpy())
        next_actions = torch.tensor(next_actions, device=device)
        next_state_values = self.target_network.get_qvalues_from_params(predicted_next_qvalue_params, next_actions)

        assert next_state_values.dim() == 1 \
              and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        # you can multiply next state values by is_not_done to achieve this.
        # TODO
        target_qvalues_for_actions = rewards + gamma * is_not_done * next_state_values

        # mean squared error loss to minimize
        # print(predicted_qvalues_for_actions, target_qvalues_for_actions)
        loss = torch.mean((predicted_qvalues_for_actions -
                        target_qvalues_for_actions.detach()) ** 2)

        if check_shapes:
            assert next_state_values.data.dim(
            ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
            assert target_qvalues_for_actions.data.dim(
            ) == 1, "there's something wrong with target q-values, they must be a vector"

        return loss

    
    def evaluate(self, env, agent, n_games=1, greedy=False, t_max=10000, visualize=False):
        """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
        rewards, step_counters = [], []
        total_counter = 0
        old_agent_epsilon = agent.epsilon
        if greedy:
            agent.epsilon = 0
        should_revert_to_train = agent.training
        agent.eval()

        for game_step in range(1, n_games + 1):
            done = False
            state, _ = env.reset()
            step_counter = 0
            while not done:
                q_value_params = agent.get_qvalue_params(np.expand_dims(state, axis=0))
                action = agent.sample_actions(q_value_params)[0]
                
                state, reward, done, _, step_counter = env.step(action)
                rewards.append(reward)
                if visualize and ((total_counter + 1) % 2 == 0):
                    print(f"State Shape: {state.shape}                  Done: {done}")
                    plt.imshow((state[:,:,-3:]).astype(int))
                    plt.show()
                    time.sleep(0.15)
                total_counter += 1
                if total_counter >= t_max:
                    return np.mean(rewards), np.mean(step_counters)
            step_counters.append(step_counter)
        if greedy:
            agent.epsilon = old_agent_epsilon 
        if should_revert_to_train:
            agent.train()
        return np.mean(rewards), np.mean(step_counters)


    def play_and_record(self, initial_state, n_steps=1):
        """
        Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
        Whenever game ends, add record with done=True and reset the game.
        It is guaranteed that env has done=False when passed to this function.

        PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

        :returns: return sum of rewards over time and the state in which the env stays
        """
        s = initial_state
        sum_rewards = 0

        should_revert_to_train = self.agent.training
        self.agent.eval()
        done = False
        # Play the game for n_steps as per instructions above
        for game_step in range(n_steps):
            # TODO: implement this section
            q_value_params = self.agent.get_qvalue_params(np.expand_dims(s, axis=0))
            action = self.agent.sample_actions(q_value_params)[0]
            next_s, reward, done, _, _ = self.env.step(action)
            
            self.exp_replay.add(s, action, reward, next_s, done)
            sum_rewards += reward
            s = next_s

            if done:
                s, _ = self.env.reset()
                done = False
        if should_revert_to_train:
            self.agent.train()
        return sum_rewards, s
    

    def train(self, timesteps_per_epoch=1, total_steps=3*10**6, save_freq=5000,
              loss_freq=150, refresh_target_network_freq=5000, eval_freq=5000,
              init_epsilon=1, final_epsilon=0.1, decay_steps=1*10**6, max_grad_norm=50,
              n_lives=5, batch_size=64, last_save_counter=None):
        # setup seed
        seed = 2
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # setup base state and environment
        state_shape = self.env.observation_space.shape
        state, _ = self.env.reset()

        # setup networks
        self.agent = self.agent.to(device)
        self.target_network = self.target_network.to(device)
        self.target_network.load_state_dict(self.agent.state_dict())

        # reset history and replay buffer
        mean_rw_history = []
        td_loss_history = []
        grad_norm_history = []
        mean_step_counts_history = []
        self.exp_replay.clear_buffer()

        # load saved model, if any
        step = 1
        if isinstance(last_save_counter, int) and last_save_counter > 0:
            step = save_freq * last_save_counter + 1
            self.agent = torch.load(f'saved-models/temporary_saved_agent_{step - 1}.pth', map_location=torch.device('cpu')).to(device)
            self.target_network.load_state_dict(self.agent.state_dict())
            self.target_network = self.target_network.to(device)

            self.opt = torch.optim.Adam(self.agent.parameters(), lr=MixedTrainer.LEARNING_RATE)
            # exp_replay.load_buffer()
            self.target_network.eval()
        
        # actually train and show results
        zero_cnt = 0
        for i in tqdm.tqdm(range(MixedTrainer.REPLAY_BUFFER_SIZE), desc="Gathering base data"):
            _, state = self.play_and_record(state, timesteps_per_epoch)
        for step in tqdm.tqdm(range(step, total_steps + 1), desc="Training Process"):
            if not is_enough_ram():
                print('less that 100 MB RAM available, freezing')
                print('make sure everything is ok and make a KeyboardInterrupt to continue')
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    pass

            self.agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

            # play
            _, state = self.play_and_record(state, timesteps_per_epoch)

            # train
            # TODO: sample batch_size of data from experience replay>
            sample_obs1, sample_actions, sample_rewards, sample_obs2, sample_dones = self.exp_replay.sample(batch_size)
            loss = self.compute_td_loss(sample_obs1, sample_actions, sample_rewards, sample_obs2, sample_dones)

            if loss == 0:
                zero_cnt += 1
            assert zero_cnt < loss_freq, "Sadge, zero loss :("

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()
            
            if (step + 1) % save_freq == 0:
                torch.save(self.agent, f'saved-models/temporary_saved_agent_{step + 1}.pth')
                # exp_replay.save_buffer()
            
            if step % loss_freq == 0:
                zero_cnt = 0
                td_loss_history.append(loss.data.cpu().item())
                grad_norm_history.append(grad_norm.cpu().item())

            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                # TODO
                self.target_network.load_state_dict(self.agent.state_dict())

            if step % eval_freq == 0:
                evaluate_results = self.evaluate(
                    self.env, self.agent, n_games=3*n_lives, greedy=True
                )
                mean_rw_history.append(evaluate_results[0])
                mean_step_counts_history.append(evaluate_results[1])
                state, _ = self.env.reset()

                print("buffer size = %i, epsilon = %.5f" % (len(self.exp_replay), self.agent.epsilon))

                plt.figure(figsize=[16, 9])

                plt.subplot(2, 2, 1)
                plt.title("Mean reward per life")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Grad norm history (smoothened)")
                plt.plot(smoothen(grad_norm_history))
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Step count history")
                plt.plot(mean_step_counts_history)
                plt.grid()

                # plt.show()
                plt.savefig("saved-models/training_stats.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Train an RL model in CARLA and plot metrics.""")
    parser.add_argument("--last_save_counter", metavar="last_save_counter", type=int, nargs='?',
                    help="Specifies the saved version of the model to use for further training.")
    args = parser.parse_args()

    print("SETTING UP ENV...")
    env = CarlaEnv(town="Town02", fps=20, im_width=1280, im_height=720, repeat_action=1, start_transform_type="random",
                   sensors="rgb", action_type="mixed", enable_preview=False, steps_per_episode=500, playing=False,
                   timeout=60)
    env = PreprocessCARLAObs(env)
    print("DONE")

    print("SETTING UP AGENT...")
    agent = MixedDQNAgent(env=env, epsilon=1)
    print("DONE")

    print("SETTING UP TARGET NETWORK...")
    target_network = MixedDQNAgent(env=env, epsilon=0).to(device)
    target_network.load_state_dict(agent.state_dict())
    print("DONE")

    print("SETTING UP TRAINER...")
    trainer = MixedTrainer(env=env, agent=agent, target_network=target_network)
    print("DONE")

    print("==================== COMMENCE TRAINING! ====================")
    trainer.train(last_save_counter=args.last_save_counter)
    print("==================== TRAINING COMPLETE! ====================")