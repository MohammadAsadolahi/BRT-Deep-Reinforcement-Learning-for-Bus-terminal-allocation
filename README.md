<div align="center">

# BRT-Deep-Reinforcement-Learning-for-Bus-terminal-allocation

### Intelligent Bus Rapid Transit Allocation via Deep Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

*A Double Deep Q-Network (DDQN) agent that learns bus dispatching strategies, reducing passenger wait times across multi-line transit networks.*

[Getting Started](#-quick-start) · [Architecture](#-architecture) · [Results](#-results) · [How It Works](#-how-it-works) · [Configuration](#%EF%B8%8F-configuration)

---

</div>

## Author

**Mohammad Asadolahi** — Senior Agentic AI Engineer

- GitHub: [https://github.com/MohammadAsadolahi](https://github.com/MohammadAsadolahi)
- Focus: Agentic AI Architectures In The Wild

---

## The Problem

Urban transit systems face a hard optimisation challenge: **given a finite fleet of buses, which line should the next bus be dispatched to — and when?**

Static schedules fail because passenger demand is stochastic, spatially uneven, and temporally varying. Over-serving one line starves another. Under-serving all lines causes cascading delays.

## The Solution

This project frames bus terminal allocation as a **Markov Decision Process** and solves it with a **Double Deep Q-Network** — a model-free reinforcement learning algorithm that:

- **Observes** passenger counts at every station and the position of every bus in the fleet.
- **Decides** which transit line to dispatch the next available bus to (or to hold).
- **Learns** from simulated interactions to maximise long-term passenger throughput while minimising system-wide wait times.

> The trained agent **outperforms a random dispatch policy** in the included comparison experiments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         System Overview                         │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Environment    │    DDQN Agent    │     Training Pipeline     │
│                  │                  │                           │
│  ┌────────────┐  │  ┌────────────┐  │  ┌─────────────────────┐  │
│  │ Passenger  │  │  │  Online    │  │  │  Experience Replay  │  │
│  │ Generation │──┼─▶│  Q-Network │──┼─▶│  Buffer (1M trans.) │  │
│  │ (Stoch.)   │  │  │  (256×256) │  │  │                     │  │
│  └────────────┘  │  └─────┬──────┘  │  └──────────┬──────────┘  │
│  ┌────────────┐  │        │         │             │             │
│  │ Multi-Line │  │  ┌─────▼──────┐  │  ┌──────────▼──────────┐  │
│  │ Bus Fleet  │◀─┼──│  ε-Greedy  │  │  │  Mini-Batch SGD     │  │
│  │ Simulation │  │  │  Policy    │  │  │  + Target Network   │  │
│  └────────────┘  │  └────────────┘  │  │  (Hard Copy @100)   │  │
│                  │                  │  └─────────────────────┘  │
└──────────────────┴──────────────────┴───────────────────────────┘
```

### Project Structure

```
BRT-Deep-Reinforcement-Learning-for-Bus-terminal-allocation/
├── src/
│   ├── __init__.py
│   ├── environment.py       # BusTransitEnvironment — Gym-style MDP
│   ├── replay_buffer.py     # Circular replay buffer
│   └── agent.py             # QNetwork + DDQNAgent (Double DQN)
├── train.py                 # CLI entry-point: train, evaluate, compare
├── Bus_Environment          # Original environment prototype
├── RL_Bus_Terminal_Allocation_DDQN.ipynb  # Jupyter notebook with experiments
├── models/                  # Saved model weights (auto-created at runtime)
├── results/                 # Training curves & plots (auto-created at runtime)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU *(optional, falls back to CPU)*

### Installation

```bash
git clone https://github.com/MohammadAsadolahi/BRT-Deep-Reinforcement-Learning-for-Bus-terminal-allocation.git
cd BRT-Deep-Reinforcement-Learning-for-Bus-terminal-allocation

pip install -r requirements.txt
```

### Train the Agent

```bash
# Full training with DDQN vs random baseline comparison
python train.py --episodes 200 --compare

# Custom configuration
python train.py --episodes 500 --buses 20 --lr 1e-4 --gamma 0.995 --seed 7
```

### Evaluate a Saved Model

```bash
python train.py --eval --model-path models/ddqn_bus.pt
```

---

## How It Works

### 1. Environment — `BusTransitEnvironment`

A simulation of a multi-line BRT network:

| Component | Description |
|-----------|-------------|
| **Stations** | Each line consists of ordered stations with independent stochastic passenger arrival rates |
| **Fleet** | A pool of `N` buses, each idle at a depot or traversing a line |
| **Passenger Arrivals** | Sampled as $\lvert \mathcal{N}(0,1) \rvert + \mu_i$ per station $i$ at each timestep |
| **Bus Movement** | Active buses advance one station per step, picking up $\min(q_i, C)$ passengers |
| **Reward Signal** | $R_t = -\sum_i q_i + 3 \sum_{b \in \text{active}} \min(q_{b}, C)$ — penalises waiting, rewards pickup |

**State vector:** $\mathbf{s} = [\underbrace{q_1, q_2, \dots, q_S}_{\text{passengers at each station}}, \underbrace{p_1, p_2, \dots, p_B}_{\text{bus positions}}]$

**Action space:** $\mathcal{A} = \{0, 1, \dots, L\}$ where $L$ = number of lines (action $L$ = hold)

### 2. Agent — Double DQN

Standard DQN overestimates Q-values because the same network both selects and evaluates actions. **Double DQN** fixes this:

$$Q_{\text{target}} = r + \gamma \cdot Q_{\theta^{-}}\!\left(s', \underset{a'}{\arg\max}\; Q_{\theta}(s', a')\right)$$

- $Q_\theta$ — **online network** (selects best action)
- $Q_{\theta^{-}}$ — **target network** (evaluates that action)
- Target network is hard-copied from the online network every 100 learning steps

### 3. Exploration — Epsilon-Greedy Decay

$$\varepsilon_{t+1} = \max(\varepsilon_t \times 0.9999,\; 0.01)$$

Starts fully exploratory ($\varepsilon = 1.0$) and anneals to 1% residual exploration.

### 4. Experience Replay

A **1M-capacity circular buffer** stores $(s, a, r, s', d)$ tuples. Random mini-batch sampling of 64 transitions per step breaks temporal correlations and stabilises learning.

---

## Results

After training, the agent produces diagnostic plots saved to `results/`:

| Plot | Description |
|------|-------------|
| `DDQN_rewards.png` | Per-episode reward with running average overlay |
| `DDQN_avg_rewards.png` | Running average reward (convergence curve) |
| `comparison.png` | Head-to-head: DDQN agent vs random dispatch policy |

---

## Configuration

All hyperparameters are tunable via CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 200 | Number of training episodes |
| `--buses` | 15 | Fleet size |
| `--batch-size` | 64 | Replay buffer sample size |
| `--lr` | 3e-4 | Adam learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--seed` | 42 | Random seed for reproducibility |
| `--compare` | off | Also run random baseline |
| `--eval` | off | Load & evaluate saved model |
| `--model-path` | `models/ddqn_bus.pt` | Path to saved weights |

### Transit Network Configuration

Modify `DEFAULT_LINE_CONFIG` in [train.py](train.py) to define custom networks:

```python
line_config = {
    "line_1": [1, 2, 3, 3, 1, 3, 4, 4, 3, 4, 2, 1],
    "line_2": [6, 7, 4, 3, 8, 9, 7, 6, 8, 9, 8, 3, 4, 4, 9, 9],
}
```

Each list represents **mean passenger arrival rates** at sequential stations along that line.

---

## Technical Highlights

- **Custom environment** — no OpenAI Gym installation required; the `BusTransitEnvironment` follows the Gym API contract (`reset()` / `step()`) for compatibility
- **Pre-allocated NumPy replay buffer** — uses pre-allocated arrays for efficient sampling at 1M capacity
- **Device-agnostic** — CPU/CUDA execution with automatic device detection
- **Reproducible** — full seed control across NumPy, Python hash, PyTorch CPU & CUDA, and cuDNN

---

this readme is AI assisted generated, so check for mistakes
