# RL-Enhanced Multi-Cloud Federation for Fault-Tolerant IoT Task Scheduling

A beginner-friendly Python simulation. It does not require cloud accounts or real IoT hardware.

## What it models
- Three simulated cloud providers with different processing speed, network latency, cost, capacity, and failure probability.
- Synthetic IoT tasks with varying size, deadline, and priority.
- A tabular Q-learning agent that learns which provider to choose.
- Retry/failover to another provider when an execution attempt fails.
- Baseline policies: round robin, least loaded, and reliability-aware.
- Evaluation metrics and plots.

## Requirements
- Python 3.10 or newer recommended
- Internet only for the first installation of Python packages
- Windows PowerShell, macOS Terminal, or Linux terminal

## Windows PowerShell: setup and run
1. Extract this ZIP.
2. Open PowerShell in the extracted `multicloud_iot_rl` folder.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
4. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run this for the current terminal only, then activate again:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run a quick test:
   ```powershell
   python main.py --episodes 30 --tasks 80 --eval-episodes 5 --out quick_run
   ```
7. Run the full demonstration:
   ```powershell
   python main.py --episodes 300 --tasks 150 --eval-episodes 30 --out results
   ```
8. Open the `results` folder. It contains:
   - `training_rewards.png`
   - `policy_comparison.png`
   - `metrics.csv`
   - `training_rewards.csv`
   - `run_summary.txt`

## Linux/macOS setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --episodes 30 --tasks 80 --eval-episodes 5 --out quick_run
python main.py --episodes 300 --tasks 150 --eval-episodes 30 --out results
```

## How to read the results
- Success rate: percentage of tasks that completed on a provider.
- Average latency: simulated elapsed time for successfully completed tasks, including failover attempts where relevant.
- Deadline miss rate: percentage of tasks that did not finish within their deadline or could not be completed.
- Average cost: simulated cost per successfully completed task.
- Recovery rate: percentage of all tasks that completed after at least one failed attempt.
- Reward: the agent's combined score based on success, priority, latency, cost, and failure/deadline penalties.

Do not claim that Q-learning is better unless the output metrics support that claim. Explain that the cloud parameters and workloads are simulated and results may change with different random seeds and assumptions.

## Suggested presentation
1. Problem: IoT devices create tasks; a single cloud can become overloaded or unavailable.
2. Proposed system: a federated pool of three simulated cloud providers.
3. RL: Q-learning learns a mapping from task/load state to cloud choice using rewards.
4. Fault tolerance: failed attempts trigger retry on another provider, with a bounded retry count.
5. Evaluation: compare Q-learning with round robin, least loaded, and reliability-aware baselines.
6. Limitations: synthetic workload, simplified cloud model, no real provider APIs, no production-grade distributed execution.

## Important implementation note
This is an academic prototype/simulation. It models provider selection and retry behavior; it is not a production cloud orchestration platform. For a stronger research study, test multiple seeds, tune the reward weights, report mean and standard deviation, and add ablation studies.
