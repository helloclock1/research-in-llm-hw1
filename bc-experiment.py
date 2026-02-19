import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Policy(nn.Module):
    def __init__(self, state_size: int, action_space_size: int, hidden_size: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        out = self.policy(x)
        return out


def collect_expert_data(expert_policy: nn.Module, num_episodes: int):
    env = gym.make("CartPole-v1")
    states = []
    expert_actions = []
    while len(states) < num_episodes:
        obs, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        while not done:
            action = expert_policy(torch.tensor(obs).float()).argmax().item()
            next_state, _, terminated, truncated, _ = env.step(action)
            episode_states.append(obs)
            episode_actions.append(action)
            obs = next_state
            done = terminated or truncated
        if len(episode_states) == 500:  # Perfect expert
            # if True:  # Imperfect expert
            states.extend(episode_states)
            expert_actions.extend(episode_actions)
    env.close()
    return states, expert_actions


plt.ion()


def train_vanilla(policy, optimizer, criterion, loader, epochs, ax):
    losses = []
    for epoch in tqdm(range(epochs), desc="Training"):
        batch_losses = []
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            logits = policy(batch_states)
            loss = criterion(logits, batch_actions)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        losses.append(sum(batch_losses) / len(batch_losses))
        ax.clear()
        ax.plot(losses)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.pause(0.01)
    return policy


def train_dagger(
    policy,
    expert_policy,
    optimizer,
    criterion,
    states,
    expert_actions,
    epochs,
    dagger_iters,
    rollouts_per_iter,
    ax,
):
    env = gym.make("CartPole-v1")
    all_states = list(states)
    all_actions = list(expert_actions)
    losses = []

    for dagger_iter in range(1, dagger_iters + 1):
        beta = 1 - dagger_iter / dagger_iters
        dataset = TensorDataset(
            torch.tensor(np.array(all_states), dtype=torch.float32),
            torch.tensor(np.array(all_actions), dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Training DAgger"):
            batch_losses = []
            for batch_states, batch_actions in loader:
                optimizer.zero_grad()
                logits = policy(batch_states)
                loss = criterion(logits, batch_actions)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            losses.append(sum(batch_losses) / len(batch_losses))
            ax.clear()
            ax.plot(losses)
            ax.set_title(f"DAgger Loss (iter {dagger_iter + 1})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            plt.pause(0.01)

        new_states = []
        new_actions = []
        for _ in range(rollouts_per_iter):
            obs, _ = env.reset()
            done = False
            while not done:
                if np.random.random() < beta:
                    action = expert_policy(torch.tensor(obs).float()).argmax().item()
                else:
                    action = policy(torch.tensor(obs).float()).argmax().item()
                new_states.append(obs)
                expert_action = expert_policy(torch.tensor(obs).float()).argmax().item()
                new_actions.append(expert_action)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        all_states.extend(new_states)
        all_actions.extend(new_actions)

    env.close()
    return policy


def evaluate_policy(env, policy, num_iters: int):
    test_rewards = []

    for _ in range(num_iters):
        episode_over = False
        rewards = []
        log_probas = []
        actions = []

        obs, info = env.reset()

        while not episode_over:
            x = torch.tensor(obs)
            logits = policy(x)

            action = logits.argmax().item()
            actions.append(action)

            log_proba = F.log_softmax(logits, dim=-1)[action].cpu()
            log_probas.append(log_proba)

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            episode_over = terminated or truncated

        test_rewards.append(sum(rewards))

    test_rewards = np.array(test_rewards)

    return test_rewards.mean(), test_rewards.std()


expert_policy = Policy(state_size=4, action_space_size=2)
expert_policy.load_state_dict(torch.load("expert_policy.pt"))


EPOCHS_PER_ITER = 20
DAGGER_ITERS = 5
ROLLOUTS_PER_ITER = 3
BC_EPOCHS = EPOCHS_PER_ITER * DAGGER_ITERS

bc_metrics = []
dagger_metrics = []

for num_episodes in [1, 2, 5, 10, 50, 100]:
    fig, ax = plt.subplots(figsize=(6, 6))
    print("Collecting expert data...")
    states, expert_actions = collect_expert_data(expert_policy, num_episodes)
    print("Collected total of", len(states), "states")
    dataset = TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(expert_actions, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    bc_policy = Policy(state_size=4, action_space_size=2)
    bc_opt = optim.AdamW(bc_policy.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    env = gym.make("CartPole-v1")

    bc_policy = train_vanilla(bc_policy, bc_opt, criterion, loader, BC_EPOCHS, ax)

    bc_eval_mean, bc_eval_std = evaluate_policy(env, bc_policy, 100)
    bc_metrics.append((bc_eval_mean, bc_eval_std))
    print(
        f"Vanilla behaviour cloning eval mean={bc_eval_mean:.4f}, eval std={bc_eval_std:.4f}"
    )

    dagger_policy = Policy(state_size=4, action_space_size=2)
    dagger_opt = optim.AdamW(dagger_policy.parameters(), lr=3e-3)
    dagger_policy = train_dagger(
        dagger_policy,
        expert_policy,
        dagger_opt,
        criterion,
        states,
        expert_actions,
        EPOCHS_PER_ITER,
        DAGGER_ITERS,
        ROLLOUTS_PER_ITER,
        ax,
    )

    dagger_eval_mean, dagger_eval_std = evaluate_policy(env, dagger_policy, 100)
    dagger_metrics.append((dagger_eval_mean, dagger_eval_std))
    print(f"DAgger eval mean={dagger_eval_mean:.4f}, eval std={dagger_eval_std:.4f}")

print(bc_metrics)
print(dagger_metrics)
