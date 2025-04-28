import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import random
import numpy as np

# ——— Hyperparams ———
ENV_ID = "Acrobot-v1"
VIDEO_DIR = "./videos"
RUN_NAME = "ppo_acrobot_gymnasium"
TIMESTEPS_PER_UPDATE = 1000
MAX_UPDATES = 1000
GAMMA = 0.99
LR = 1e-3
EPS_CLIP = 0.2
K_EPOCHS = 8
DEVICE = "cuda"
SAVE_INTERVAL = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ——— Actor-Critic Network ———
class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        self.to(DEVICE)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def act(self, state, memory=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        a_tensor = dist.sample()
        action = a_tensor.item()
        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(a_tensor).item())
        return action

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, values


# ——— Rollout Buffer ———
class Memory:

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# ——— PPO Agent ———
class PPO:

    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.policy.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.to(DEVICE)
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of returns
        returns = []
        discounted = 0
        for reward, is_term in zip(reversed(memory.rewards),
                                   reversed(memory.is_terminals)):
            discounted = 0 if is_term else discounted
            discounted = reward + GAMMA * discounted
            returns.insert(0, discounted)
        # to tensor [T,1] and normalize
        returns = torch.tensor(returns,
                               dtype=torch.float32).unsqueeze(1).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Convert memory to tensors
        old_states = torch.cat(memory.states, dim=0).detach().to(DEVICE)
        old_actions = torch.tensor(memory.actions,
                                   dtype=torch.long).detach().to(DEVICE)
        old_logprobs = torch.tensor(memory.logprobs,
                                    dtype=torch.float32).detach().to(DEVICE)

        # PPO updates
        for _ in range(K_EPOCHS):
            logprobs, entropy, state_values = self.policy.evaluate(
                old_states, old_actions)
            advantages = returns.squeeze(1) - state_values.squeeze(1)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP,
                                1 + EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2).mean() \
                   + 0.5 * self.MseLoss(state_values, returns) \
                   - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Sync old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


# ——— Main ———
def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # record only the very first episode
    env = RecordVideo(env,
                      video_folder=VIDEO_DIR,
                      episode_trigger=lambda ep: (ep % SAVE_INTERVAL == 0))
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}")

    total_timesteps = 0
    for update in range(1, MAX_UPDATES + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = ppo.policy_old.act(state, memory)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            ep_reward += reward
            state = next_state
            total_timesteps += 1

            if total_timesteps >= TIMESTEPS_PER_UPDATE:
                ppo.update(memory)
                memory.clear()
                total_timesteps = 0

        writer.add_scalar("Episode Reward", ep_reward, update)
        print(f"Update {update:3d}\tEpisode Reward: {ep_reward:.2f}")

    writer.close()
    env.close()
    print(f"\nTraining finished. Video of the first episode is in {VIDEO_DIR}")


if __name__ == "__main__":
    main()
