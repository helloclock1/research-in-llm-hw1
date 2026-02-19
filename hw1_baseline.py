# %%
import copy
import datetime

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


# %%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
obs, info = env.reset()
print(f"{obs}; {info}")

# %%
state_space_size = len(obs)
action_space_size = env.action_space.n

# %%
policy = Policy(state_space_size, action_space_size, 128)
policy.to(device)

# %%
EPOCHS = 2000
# EPOCHS = 50

# optimizer = optim.SGD(policy.parameters(), lr=3e-2)
optimizer = optim.AdamW(policy.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# %%
def compute_return(rewards: np.ndarray, gamma: float, t: int):
    T = rewards.shape[0]
    K = T - t
    coeffs = np.cumprod([1] + [gamma] * (K - 1))
    return_value = (rewards[t:] * coeffs).sum()
    return return_value


# %%
GAMMA = 0.995

rewards_total = []

plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))

best_policy = None
best_avg = -float("inf")

for epoch in range(EPOCHS):
    episode_over = False
    rewards = []
    log_probas = []
    actions = []

    obs, info = env.reset()

    while not episode_over:
        x = torch.tensor(obs).to(device)
        logits = policy(x)
        # proba = F.softmax(logits, dim=-1).detach().cpu().numpy()

        dist = torch.distributions.Categorical(logits=logits)
        # action = np.random.choice(action_space_size, p=proba)
        action = dist.sample()
        actions.append(action)

        # log_proba = F.log_softmax(logits, dim=-1)[action].cpu()
        # log_probas.append(log_proba)
        log_probas.append(dist.log_prob(action))

        obs, reward, terminated, truncated, info = env.step(action.item())
        rewards.append(reward)

        episode_over = terminated or truncated

    timesteps = len(rewards)
    rewards = np.array(rewards)

    returns = []
    for t in range(timesteps):
        R_t = compute_return(rewards, GAMMA, t)
        returns.append(R_t)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probas = torch.stack(log_probas)

    optimizer.zero_grad()
    # loss = -(log_probas * returns).mean()
    # loss = -(log_probas * returns.to(device)).sum()
    loss = -(log_probas * returns.to(device)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    # scheduler.step()

    episode_return = returns[0].item()

    # print(f'Episode #{epoch} finished with reward={sum(rewards)}, return={episode_return:.2f}, loss={loss.item():.2f}')

    rewards_total.append(sum(rewards))

    if epoch % 10 == 0:
        ax.cla()
        ax.plot(rewards_total, alpha=0.3, label="Raw rewards")
        window_size = 50
        moving_avg = np.convolve(
            rewards_total, np.ones(window_size) / window_size, mode="valid"
        )
        ax.plot(
            moving_avg,
            linewidth=2,
            label=f"{window_size} episode average",
        )
        ax.axhline(y=195, color="r", linestyle="--", label="Solved threshold")
        ax.legend(loc="upper left")
        fig.canvas.draw()
        plt.pause(0.001)

        if moving_avg[-1] > best_avg:
            best_policy = copy.deepcopy(policy)

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
        x = torch.tensor(obs).to(device)
        logits = best_policy(x)
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
