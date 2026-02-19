# %%
import copy
import datetime

import gymnasium as gym
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
# env = gym.make('CartPole-v1', render_mode='human')
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
TOTAL_EPISODES = 2000
ROLLOUTS = 4
EPOCHS = TOTAL_EPISODES // ROLLOUTS

# optimizer = optim.SGD(policy.parameters(), lr=1e-1)
optimizer = optim.AdamW(policy.parameters(), lr=3e-3)
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

plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))

best_policy = None
best_avg = -float("inf")

for epoch in range(EPOCHS):
    returns_total = []
    lengths_total = []
    log_probas_total = []
    entropies_total = []
    raw_rewards = []

    for rollout in range(ROLLOUTS):
        episode_over = False
        rewards = []
        log_probas = []
        actions = []
        entropies = []
        obs, info = env.reset()

        while not episode_over:
            x = torch.tensor(obs).to(device)
            logits = policy(x)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            actions.append(action)

            log_probas.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)

            episode_over = terminated or truncated

        timesteps = len(rewards)
        rewards = np.array(rewards)
        raw_rewards.append(sum(rewards))

        returns = []
        for t in range(timesteps):
            R_t = compute_return(rewards, GAMMA, t)
            returns.append(R_t)

        returns = torch.tensor(returns).to(device)
        log_probas = torch.stack(log_probas)
        entropies = torch.stack(entropies)

        returns_total.append(returns)
        log_probas_total.append(log_probas)
        lengths_total.append(timesteps)
        entropies_total.append(entropies)

    optimizer.zero_grad()
    episode_returns = torch.tensor([r[0].item() for r in returns_total])
    entropy_coeff = 1e-2
    loss = torch.tensor(0.0, device=device)
    for k in range(ROLLOUTS):
        baseline = (episode_returns.sum() - episode_returns[k]) / (ROLLOUTS - 1)
        advantages = returns_total[k] - baseline.to(device)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss -= (log_probas_total[k] * advantages).mean()
        loss -= entropy_coeff * entropies_total[k].mean()
    loss /= ROLLOUTS
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    # rewards_total.append(episode_returns.mean().item())
    for k in range(ROLLOUTS):
        # rewards_total.append(episode_returns[k].item())
        rewards_total.append(raw_rewards[k])

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
            best_avg = moving_avg[-1]
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
