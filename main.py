import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Cloud:
    name: str
    speed: float
    latency: float
    cost_per_unit: float
    failure_prob: float
    capacity: float


# Illustrative simulation parameters, not real AWS/Azure/GCP measurements.
CLOUDS = [
    Cloud("Cloud A", speed=8.0, latency=1.2, cost_per_unit=0.90, failure_prob=0.08, capacity=90.0),
    Cloud("Cloud B", speed=11.0, latency=2.0, cost_per_unit=1.15, failure_prob=0.04, capacity=120.0),
    Cloud("Cloud C", speed=6.5, latency=0.7, cost_per_unit=0.12, failure_prob=0.12, capacity=75.0),
]

DRAIN_FACTOR = 0.72  # workload drainage between task arrivals


@dataclass(frozen=True)
class Task:
    size: float
    deadline: float
    priority: int


def make_tasks(count, rng):
    tasks = []
    for _ in range(count):
        size = rng.uniform(8.0, 55.0)
        deadline = rng.uniform(4.0, 13.0) + size / 15.0
        priority = rng.choices([1, 2, 3], weights=[0.50, 0.32, 0.18], k=1)[0]
        tasks.append(Task(float(size), float(deadline), int(priority)))
    return tasks


def estimate_time(task, cloud, load):
    queue_factor = 1.0 + min(max(load, 0.0) / max(cloud.capacity, 1e-9), 2.0) * 0.75
    return cloud.latency + (task.size / cloud.speed) * queue_factor


def state_for(task, loads):
    """Discrete state includes task size, deadline slack, priority and cloud loads."""
    size_bin = min(3, int(task.size / 14.0))
    fastest_estimate = min(estimate_time(task, cloud, load) for cloud, load in zip(CLOUDS, loads))
    slack = task.deadline - fastest_estimate
    slack_bin = 0 if slack < 0 else (1 if slack < 1.5 else (2 if slack < 4.0 else 3))
    priority_bin = task.priority - 1
    load_bins = tuple(min(4, int(5 * max(0.0, load) / cloud.capacity)) for load, cloud in zip(loads, CLOUDS))
    return (size_bin, slack_bin, priority_bin) + load_bins


def feasible_clouds(task, loads):
    """Clouds predicted to meet the deadline; if none can, return all clouds."""
    feasible = [i for i, cloud in enumerate(CLOUDS)
                if estimate_time(task, cloud, loads[i]) <= task.deadline]
    return feasible if feasible else list(range(len(CLOUDS)))


class QAgent:
    def __init__(self, alpha=0.12, gamma=0.85, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.992, seed=7):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

    def row(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(len(CLOUDS), dtype=float)
        return self.q[state]

    def choose(self, state, allowed_actions, explore=True):
        allowed = list(allowed_actions)
        if not allowed:
            allowed = list(range(len(CLOUDS)))
        values = self.row(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(allowed)
        best_value = max(values[a] for a in allowed)
        best = [a for a in allowed if values[a] == best_value]
        return self.rng.choice(best)

    def update(self, state, action, reward, next_state, next_allowed, terminal=False):
        current = self.row(state)[action]
        future = 0.0 if terminal else max(self.row(next_state)[a] for a in next_allowed)
        target = reward + self.gamma * future
        self.q[state][action] = current + self.alpha * (target - current)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def select_action(policy, task, loads, state, agent, explore, task_index, policy_rng):
    estimates = [estimate_time(task, cloud, loads[i]) for i, cloud in enumerate(CLOUDS)]
    if policy == "q_learning":
        # Deadline-aware action masking: prefer clouds predicted to meet deadline.
        return agent.choose(state, feasible_clouds(task, loads), explore=explore)
    if policy == "random_feasible":
        # Same masking as Q-learning but no learning (ablation baseline).
        return policy_rng.choice(feasible_clouds(task, loads))
    if policy == "cheapest_feasible":
        # Cheapest cloud among those predicted to meet the deadline.
        feas = feasible_clouds(task, loads)
        return min(feas, key=lambda i: (CLOUDS[i].cost_per_unit, estimates[i]))
    if policy == "round_robin":
        return task_index % len(CLOUDS)
    if policy == "least_loaded":
        return int(np.argmin(loads))
    if policy == "reliability_aware":
        scores = [estimates[i] + CLOUDS[i].failure_prob * 5.0 for i in range(len(CLOUDS))]
        return int(np.argmin(scores))
    raise ValueError(f"Unknown policy: {policy}")


def run_episode(tasks, policy, agent=None, explore=False, seed=1, max_retries=2, learn=False):
    rng = random.Random(seed)
    policy_rng = random.Random(seed + 7919)
    loads = [0.0] * len(CLOUDS)
    totals = {
        "reward": 0.0, "completed": 0, "deadline_misses": 0,
        "attempt_failures": 0, "recovered": 0, "latency_sum": 0.0,
        "cost_sum": 0.0, "task_count": len(tasks), "on_time": 0,
    }

    for idx, task in enumerate(tasks):
        state = state_for(task, loads)
        chosen = select_action(policy, task, loads, state, agent, explore, idx, policy_rng)
        # Try the selected cloud first, then fail over to other providers ordered by
        # estimated completion time. Maximum attempts = initial attempt + retries.
        remaining = [i for i in range(len(CLOUDS)) if i != chosen]
        remaining.sort(key=lambda i: estimate_time(task, CLOUDS[i], loads[i]) + CLOUDS[i].failure_prob * 5.0)
        order = [chosen] + remaining
        completed = False
        recovered = False
        elapsed = 0.0
        task_cost = 0.0
        final_latency = 0.0
        attempts = 0

        for cloud_idx in order:
            if attempts >= max_retries + 1:
                break
            cloud = CLOUDS[cloud_idx]
            attempts += 1
            attempt_time = estimate_time(task, cloud, loads[cloud_idx])
            elapsed += attempt_time
            task_cost += task.size * cloud.cost_per_unit
            loads[cloud_idx] += task.size * 0.30
            if rng.random() >= cloud.failure_prob:
                completed = True
                recovered = attempts > 1
                final_latency = elapsed
                break
            totals["attempt_failures"] += 1

        # Count all expenditure, including work spent on failed attempts.
        totals["cost_sum"] += task_cost
        on_time = completed and final_latency <= task.deadline
        if completed:
            totals["completed"] += 1
            totals["latency_sum"] += final_latency
            if recovered:
                totals["recovered"] += 1
            if on_time:
                totals["on_time"] += 1
                reward = 18.0 + 2.0 * task.priority - 0.55 * final_latency - 0.12 * task_cost
            else:
                totals["deadline_misses"] += 1
                lateness = max(0.0, final_latency - task.deadline)
                reward = -18.0 - 2.0 * task.priority - 2.0 * lateness - 0.05 * task_cost
        else:
            totals["deadline_misses"] += 1
            reward = -28.0 - 2.0 * task.priority - 0.05 * task_cost

        totals["reward"] += reward

        # FIX 3: apply workload drainage BEFORE computing the next state, so the
        # next state is exactly what the agent will observe on the next arrival.
        loads = [max(0.0, load * DRAIN_FACTOR) for load in loads]

        # FIX 1: only update the Q-table when learning is explicitly enabled.
        if policy == "q_learning" and agent is not None and learn:
            terminal = idx == len(tasks) - 1
            if terminal:
                next_state, next_allowed = state, [0]
            else:
                next_task = tasks[idx + 1]
                next_state = state_for(next_task, loads)
                next_allowed = feasible_clouds(next_task, loads)
            agent.update(state, chosen, reward, next_state, next_allowed, terminal=terminal)

    n = max(1, totals["task_count"])
    completed = totals["completed"]
    totals["success_rate"] = 100.0 * completed / n
    totals["deadline_miss_rate"] = 100.0 * totals["deadline_misses"] / n
    totals["deadline_hit_rate"] = 100.0 * totals["on_time"] / n
    totals["failure_attempts_per_task"] = totals["attempt_failures"] / n
    totals["recovery_rate"] = 100.0 * totals["recovered"] / n
    totals["avg_latency"] = totals["latency_sum"] / max(1, completed)
    totals["avg_cost"] = totals["cost_sum"] / max(1, completed)
    totals["avg_reward"] = totals["reward"] / n
    return totals


def moving_average(values, window=20):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_plots(out_dir, training_rewards, results):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(training_rewards) + 1), training_rewards, alpha=0.30, label="Episode reward")
    window = min(20, max(1, len(training_rewards)))
    ma = moving_average(training_rewards, window)
    x = np.arange(len(ma)) + window
    plt.plot(x, ma, linewidth=2, label=f"Moving average ({window})")
    plt.title("Q-learning training reward")
    plt.xlabel("Training episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_rewards.png", dpi=160)
    plt.close()

    labels = list(results.keys())
    metrics = [
        ("deadline_hit_rate", "On-time rate (%)"),
        ("avg_latency", "Average latency of completed tasks"),
        ("deadline_miss_rate", "Deadline miss rate (%)"),
        ("avg_cost", "Average cost per completed task"),
        ("avg_reward", "Average reward per task"),
        ("recovery_rate", "Tasks recovered after a failure (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        ax.bar(labels, [results[name][key] for name in labels])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Scheduling policy comparison (evaluation episodes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "policy_comparison.png", dpi=160)
    plt.close(fig)


METRIC_KEYS = ["success_rate", "deadline_hit_rate", "deadline_miss_rate", "failure_attempts_per_task",
               "recovery_rate", "avg_latency", "avg_cost", "avg_reward"]


def mean_metrics(records):
    return {key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS}


def std_metrics(records):
    return {key: float(np.std([r[key] for r in records], ddof=1)) if len(records) > 1 else 0.0
            for key in METRIC_KEYS}


def main():
    parser = argparse.ArgumentParser(description="Deadline-aware Q-learning for simulated multi-cloud IoT scheduling")
    parser.add_argument("--episodes", type=int, default=300, help="Q-learning training episodes")
    parser.add_argument("--tasks", type=int, default=150, help="IoT tasks per episode")
    parser.add_argument("--eval-episodes", type=int, default=30, help="evaluation episodes per policy")
    parser.add_argument("--out", type=str, default="results_v3", help="output folder")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    if min(args.episodes, args.tasks, args.eval_episodes) < 1:
        parser.error("--episodes, --tasks and --eval-episodes must all be positive integers")
    if args.eval_episodes >= 5000:
        parser.error("--eval-episodes must be below 5000")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)
    agent = QAgent(seed=args.seed)
    training_rewards = []

    print("=== Training deadline-aware Q-learning scheduler ===")
    for ep in range(1, args.episodes + 1):
        tasks = make_tasks(args.tasks, random.Random(master_rng.randrange(1_000_000_000)))
        stats = run_episode(tasks, "q_learning", agent=agent, explore=True,
                            seed=master_rng.randrange(1_000_000_000), learn=True)
        training_rewards.append(stats["reward"])
        agent.decay_exploration()
        if ep == 1 or ep % max(1, args.episodes // 10) == 0:
            print(f"episode {ep:4d}/{args.episodes} | reward={stats['reward']:8.2f} "
                  f"| completion={stats['success_rate']:6.2f}% | on-time={stats['deadline_hit_rate']:6.2f}% "
                  f"| epsilon={agent.epsilon:.3f}")

    policies = ["q_learning", "random_feasible", "cheapest_feasible",
                "round_robin", "least_loaded", "reliability_aware"]
    names = {"q_learning": "Deadline-aware Q-learning",
             "random_feasible": "Random feasible (no learning)",
             "cheapest_feasible": "Cheapest feasible",
             "round_robin": "Round robin",
             "least_loaded": "Least loaded",
             "reliability_aware": "Reliability-aware"}
    records = {names[p]: [] for p in policies}

    # FIX 2: evaluation seeds are derived from a per-run block so different
    # --seed values never share evaluation episodes (valid while seeds differ).
    eval_base = 1_000_000 + args.seed * 10_000

    print("\n=== Evaluation (exploration disabled, learning disabled) ===")
    for ep in range(args.eval_episodes):
        tasks = make_tasks(args.tasks, random.Random(eval_base + ep))
        # Same task sequence and random stream seed for all policies in each episode.
        for p in policies:
            stats = run_episode(tasks, p, agent=agent if p == "q_learning" else None,
                                explore=False, seed=eval_base + 5_000 + ep, learn=False)
            records[names[p]].append(stats)

    results = {name: mean_metrics(rows) for name, rows in records.items()}
    stds = {name: std_metrics(rows) for name, rows in records.items()}
    for name, m in results.items():
        s = stds[name]
        print(f"{name:30s} | completion={m['success_rate']:6.2f}% | on-time={m['deadline_hit_rate']:6.2f}% "
              f"(±{s['deadline_hit_rate']:.2f}) | misses={m['deadline_miss_rate']:6.2f}% "
              f"| latency={m['avg_latency']:6.2f} | cost={m['avg_cost']:8.2f} "
              f"| recovery={m['recovery_rate']:6.2f}% | reward={m['avg_reward']:8.2f} (±{s['avg_reward']:.2f})")

    save_plots(out_dir, training_rewards, results)
    metric_keys = list(next(iter(results.values())).keys())
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy"] + metric_keys)
        for name, metrics in results.items():
            writer.writerow([name] + [f"{metrics[k]:.4f}" for k in metric_keys])
    with (out_dir / "training_rewards.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])
        writer.writerows((i + 1, f"{r:.4f}") for i, r in enumerate(training_rewards))
    with (out_dir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Deadline-aware RL-Enhanced Multi-Cloud Federation for Fault-Tolerant IoT Task Scheduling\n")
        f.write(f"Training episodes: {args.episodes}\nTasks per episode: {args.tasks}\nEvaluation episodes per policy: {args.eval_episodes}\nSeed: {args.seed}\n\n")
        for name, metrics in results.items():
            f.write(f"{name}\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
    print(f"\nFiles saved in: {out_dir.resolve()}")
    print("  training_rewards.png\n  policy_comparison.png\n  metrics.csv\n  training_rewards.csv\n  run_summary.txt")
    print("\nImportant: all clouds, task loads, failures, costs, and latencies are simulated; this is not a live cloud deployment.")


if __name__ == "__main__":
    main()