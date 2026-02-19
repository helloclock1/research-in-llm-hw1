# %%
import copy
import datetime
import random
from multiprocessing import Value
from tarfile import DIRTYPE

import gymnasium as gym
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# %%
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device}")


# %%
class Policy(nn.Module):
    def __init__(self, state_size: int, action_space_size: int, hidden_size: int):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, action_space_size),
        )

    def forward(self, x: torch.Tensor):
        out = self.policy(x)
        return out


class ValueFunction(nn.Module):
    def __init__(self, state_size: int, hidden_size: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor):
        out = self.value(x).squeeze(-1)
        return out


# %%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
obs, info = env.reset()
print(f"{obs}; {info}")

# %%
state_space_size = len(obs)
action_space_size = env.action_space.n

# %%
policy_fn = Policy(state_space_size, action_space_size, 128)
policy_fn.to(device)

value_fn = ValueFunction(state_space_size, 128)
value_fn.to(device)

# %%
EPOCHS = 2000
# EPOCHS = 500

# optimizer = optim.SGD(policy.parameters(), lr=3e-2)
policy_opt = optim.AdamW(policy_fn.parameters(), lr=1e-3)
value_opt = optim.AdamW(value_fn.parameters(), lr=3e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# %%
def compute_return(rewards: np.ndarray, gamma: float, t: int):
    T = rewards.shape[0]
    K = T - t
    coeffs = np.cumprod([1] + [gamma] * (K - 1))
    return_value = (rewards[t:] * coeffs).sum()
    return return_value


# %%
GAMMA = 0.99

rewards_total = []
value_losses_total = []

plt.ion()
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

best_policy_fn = None
best_value_fn = None
best_avg = -float("inf")

for epoch in range(EPOCHS):
    episode_over = False
    rewards = []
    log_probas = []
    states = []
    actions = []

    obs, info = env.reset()

    while not episode_over:
        x = torch.tensor(obs, dtype=torch.float32).to(device)
        states.append(x)
        logits = policy_fn(x)
        proba = F.softmax(logits, dim=-1).detach().cpu().numpy()

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        actions.append(action.item())

        log_proba = dist.log_prob(action)
        log_probas.append(log_proba)

        obs, reward, terminated, truncated, info = env.step(action.item())
        rewards.append(reward)

        episode_over = terminated or truncated

    timesteps = len(rewards)
    rewards = np.array(rewards)

    returns = []
    for t in range(timesteps):
        R_t = compute_return(rewards, GAMMA, t)
        returns.append(R_t)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    log_probas = torch.stack(log_probas)

    value_opt.zero_grad()

    states = torch.stack(states)
    state_values = value_fn(states)
    value_loss = F.mse_loss(state_values, returns)

    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=1.0)
    value_opt.step()

    with torch.no_grad():
        state_values = value_fn(states)
        advantages = returns - state_values
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(
            device
        )

    policy_opt.zero_grad()
    # policy_loss = -(log_probas.cpu() * advantages.cpu()).sum()
    policy_loss = -(log_probas.cpu() * advantages.cpu()).mean()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_fn.parameters(), max_norm=1.0)
    policy_opt.step()

    episode_return = returns[0].item()

    rewards_total.append(sum(rewards))
    value_losses_total.append(value_loss.item())

    if epoch % 10 == 0:
        ax[0].cla()
        ax[0].plot(rewards_total, alpha=0.3, label="Raw rewards")
        window_size = 50
        moving_avg = np.convolve(
            rewards_total, np.ones(window_size) / window_size, mode="valid"
        )
        ax[0].plot(
            moving_avg,
            linewidth=2,
            label=f"{window_size} episode average",
        )
        ax[0].axhline(y=195, color="r", linestyle="--", label="Solved threshold")
        ax[0].legend(loc="upper left")

        ax[1].cla()
        ax[1].plot(value_losses_total, alpha=0.7, label="Value loss")
        ax[1].set_title("Value function loss")
        ax[1].legend()

        fig.canvas.draw()
        plt.pause(0.001)

        if moving_avg[-1] > best_avg:
            best_avg = moving_avg[-1]
            best_policy_fn = copy.deepcopy(policy_fn)
            best_value_fn = copy.deepcopy(value_fn)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig.savefig(f"rewards_{timestamp}.png", dpi=300, bbox_inches="tight")

test_rewards = []

for _ in range(100):
    episode_over = False
    rewards = []
    log_probas = []
    actions = []

    obs, info = env.reset()

    while not episode_over:
        x = torch.tensor(obs, dtype=torch.float32).to(device)
        logits = best_policy_fn(x)
        proba = F.softmax(logits, dim=-1).detach().cpu().numpy()

        # action = np.random.choice(action_space_size, p=proba)
        action = logits.argmax().item()
        actions.append(action)

        log_proba = F.log_softmax(logits, dim=-1)[action].cpu()
        log_probas.append(log_proba)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        episode_over = terminated or truncated

    test_rewards.append(sum(rewards))

test_rewards = np.array(test_rewards)

print(f"Test mean={test_rewards.mean():.4f}, std={test_rewards.std():.4f}")
