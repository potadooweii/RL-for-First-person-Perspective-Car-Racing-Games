import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rainbow_DQN.replay_buffer import *
from rainbow_DQN.rainbow_dqn import DQN
import argparse
from environment_wrapper.racecar_env import CarRacingEnvironment


class Runner:
    def __init__(self, args, seed=123):
        self.args = args
        self.seed = seed

        self.env = CarRacingEnvironment(args.scenario, test=False, continuous_action=False)
        self.env_evaluate = CarRacingEnvironment(args.scenario, test=True, continuous_action=False)  # When evaluating the policy, we need to rebuild an environment
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.args.state_dim = self.env.observation_shape
        self.args.action_dim = self.env.action_space.shape[0]
        print("env={}".format(self.args.exp_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        self.writer = SummaryWriter(log_dir=f'/disk1/nfs/bwdong/rl_final/log/ver3/{args.exp_name}')

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        episode_idx = 0
        while self.total_steps < self.args.max_train_steps:

            state, info = self.env.reset()

            done = False
            episode_steps = 0
            episode_reward = 0
            episode_idx += 1

            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done, truncate, info = self.env.step(action)
                episode_steps += 1
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                episode_reward += reward

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done or truncate:
                    self.writer.add_scalar(f'Train/Episode Reward', episode_reward, self.total_steps)
                    self.writer.add_scalar(f'Train/Episode Len', episode_steps, self.total_steps)
                    print(f"[{self.total_steps}/{self.args.max_train_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_steps}")
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal, done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

            if episode_idx % self.args.evaluate_freq == 0:
                self.evaluate_policy()
        # Save reward
        # np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        self.agent.net.eval()
        for _ in range(self.args.evaluate_times):
            state, info = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, done, truncate, info = self.env_evaluate.step(action)
                done = done or truncate
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward
        self.agent.net.train()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar(f'Evaluate/Episode Reward', evaluate_reward, self.total_steps)
        torch.save(self.agent.net.state_dict(), f'/disk1/nfs/bwdong/rl_final/log/ver3/{args.exp_name}/model_{self.total_steps}_{round(evaluate_reward,2)}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--exp_name", type=str, default="DQN")
    parser.add_argument("--scenario", type=str, default="austria_competition")
    parser.add_argument("--max_train_steps", type=int, default=int(1e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=10, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(2**13), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()
    runner = Runner(args=args)
    runner.run()
