# Reinforcement Learning for a Two-Product Inventory System

## Overview

This project investigates the application of reinforcement learning to the replenishment management of a warehouse that handles two different products.

At the beginning of each simulated day, the warehouse manager must decide:

1. whether to place a replenishment order for each product;
2. how many discrete units to order.

The objective is to minimize the total inventory-management cost, including:

- ordering costs;
- inventory holding costs;
- shortage costs.

The project implements and evaluates two reinforcement learning models:

- Double Deep Q-Network, or Double DQN;
- Actor-Critic.

Their performance is compared with a traditional periodic-review `(s, S)` replenishment policy, also referred to as the `s-min/S-max` policy.

---

## Assignment

The warehouse stores two different products, each obtained from a different supplier.

At the beginning of each day, a replenishment decision must be made for each product. If an order is placed, its quantity must be expressed as an integer number of units.

At least one reinforcement learning agent must be used to manage the replenishment policy.

The performance of the reinforcement learning solution must be compared with the performance of an `s-min/S-max` policy for each product.

The optimization objective is to minimize the overall expected cost of operating the warehouse.

---

## Simulation Parameters

### General Cost Parameters

| Parameter | Description | Value |
|---|---|---:|
| `K` | Fixed setup cost for a positive order | 10 |
| `i` | Incremental cost per ordered unit | 3 |
| `h` | Inventory holding cost per unit per unit of time | 1 |
| `π` | Shortage cost per unit per unit of time | 7 |

No fixed setup cost is charged when the order quantity is zero.

### Demand Process

Demand arrivals follow an exponential inter-arrival-time distribution with rate:

```text
lambda = 0.1
```

Therefore, the expected inter-arrival time is:

```text
E[T] = 1 / lambda = 10
```

Each arrival requests a random discrete quantity according to the demand distribution of the corresponding product.

---

## Product 1

### Demand Size Distribution

| Demand size | Probability |
|---:|---:|
| 1 | 1/6 |
| 2 | 1/3 |
| 3 | 1/3 |
| 4 | 1/6 |

### Lead Time

```text
Uniform(0.5, 1.0)
```

### Benchmark Policy Parameters

```text
s = 0
S = 20
```

---

## Product 2

### Demand Size Distribution

| Demand size | Probability |
|---:|---:|
| 5 | 1/8 |
| 4 | 1/2 |
| 3 | 1/4 |
| 2 | 1/8 |

### Lead Time

```text
Uniform(0.2, 0.7)
```

### Benchmark Policy Parameters

```text
s = 5
S = 20
```

---

## Simulation Horizon

Each episode represents:

```text
150 simulated days
```

One replenishment decision is made at the beginning of each simulated day.

Demand arrivals and replenishment deliveries can occur at continuous times during the day because the inventory system is simulated using SimPy.

---

# Implementation

## Notebook Structure

The notebook is divided into three main cells.

### Cell 1: Simulation, Double DQN, Benchmark, and Utilities

The first cell contains:

- package installation and imports;
- global simulation parameters;
- action encoding and decoding functions;
- the product simulation models;
- the warehouse environment used by the reinforcement learning agents;
- the Double DQN neural network;
- action selection and Double DQN optimization;
- Double DQN training and evaluation functions;
- the `(s, S)` benchmark environment;
- Welch warm-up analysis;
- statistical comparison functions.

### Cell 2: Double DQN Experiment

The second cell contains:

- Double DQN training;
- Double DQN evaluation;
- execution of the `(s, S)` benchmark;
- confidence-interval estimation;
- paired comparisons of cost components;
- analysis of the actions selected by the agent;
- plots of training and evaluation results.

### Cell 3: Actor-Critic Experiment

The third cell contains:

- the Actor-Critic neural network;
- state preprocessing;
- stochastic and deterministic action selection;
- Actor-Critic training;
- Actor-Critic evaluation;
- comparison with the `(s, S)` benchmark;
- analysis of the selected actions;
- training and evaluation plots.

---

## Discrete-Event Simulation

The inventory system is implemented with SimPy.

The simulation includes two main types of asynchronous events:

1. demand arrivals;
2. replenishment-order arrivals.

Demand inter-arrival times are generated using:

```python
random.expovariate(0.1)
```

The parameter passed to `expovariate` is the exponential rate `lambda`, not the mean inter-arrival time.

Order lead times are sampled from the uniform distribution associated with each product.

The inventory level is allowed to become negative. A negative inventory value represents backlogged demand rather than lost sales.

---

## State Representation

The reinforcement learning agents observe the following state:

```text
[
    inventory level of product 1,
    inventory level of product 2,
    units of product 1 already on order,
    units of product 2 already on order
]
```

The `on_order` variables represent units that have been ordered but have not yet arrived.

Including these variables is important because two states with the same on-hand inventory can require different replenishment decisions.

For example:

```text
inventory = 5, on_order = 0
```

is different from:

```text
inventory = 5, on_order = 20
```

Without the `on_order` variables, the agent would not be able to distinguish between these situations.

---

## State Normalization

Before a state is passed to a neural network, it is divided by a scaling constant:

```python
STATE_SCALE = 50.0
```

For example, the state:

```text
[20, -5, 30, 10]
```

is transformed into:

```text
[0.4, -0.1, 0.6, 0.2]
```

This transformation does not change the warehouse simulation. It only improves the numerical scale of the neural-network inputs and can make training more stable.

---

## Action Representation

The maximum quantity that can be ordered for each product is defined globally:

```python
MAX_ORDER_QTY = 30
```

Each product can therefore receive an order belonging to:

```text
{0, 1, 2, ..., 30}
```

The value `0` represents the decision not to place an order.

The number of possible order quantities for one product is:

```python
ACTION_SIZE = MAX_ORDER_QTY + 1
```

With `MAX_ORDER_QTY = 30`:

```text
ACTION_SIZE = 31
```

The custom Double DQN and Actor-Critic agents represent the two product decisions as one combined discrete action:

```python
TOTAL_ACTIONS = ACTION_SIZE * ACTION_SIZE
```

Therefore:

```text
TOTAL_ACTIONS = 31 * 31 = 961
```

The combined action is decoded as follows:

```python
order_product_1 = action // ACTION_SIZE
order_product_2 = action % ACTION_SIZE
```

For example, the order pair `(10, 20)` is encoded as:

```python
action = 10 * 31 + 20
```

The use of global constants prevents inconsistencies between the maximum order quantity and the neural-network output dimension.

---

## Replenishment Mechanism

When an agent selects a positive order quantity, the following sequence occurs:

1. the order quantity is immediately added to `on_order`;
2. the fixed and variable ordering costs are charged;
3. the lead time is sampled;
4. after the lead time, the quantity is added to the on-hand inventory;
5. the same quantity is removed from `on_order`.

The implementation follows this logic when an order is created:

```python
on_order += quantity
```

When the order arrives:

```python
inventory_level += quantity
on_order -= quantity
```

If the selected quantity is zero, no order is created and no setup cost is charged.

Multiple orders can be in transit at the same time. The `on_order` variable stores the total quantity of all outstanding orders.

---

## Cost Calculation

The total system cost is defined as:

```text
total cost =
    ordering cost
    + holding cost
    + shortage cost
```

### Ordering Cost

For a positive order quantity `q`:

```text
ordering cost = K + i * q
```

Using the assignment parameters:

```text
ordering cost = 10 + 3 * q
```

For `q = 0`, the ordering cost is zero.

### Holding Cost

When the inventory level is positive, the holding cost is accumulated continuously:

```text
holding cost =
    h * positive inventory * duration
```

### Shortage Cost

When the inventory level is negative, the shortage cost is accumulated continuously:

```text
shortage cost =
    pi * absolute negative inventory * duration
```

The implementation also accounts for the interval between the most recent inventory change and the current simulation time.

---

## Reward Function

The reinforcement learning objective is defined using the negative daily cost as the reward:

```text
reward = -daily total cost
```

Reinforcement learning algorithms maximize expected cumulative reward. Therefore, maximizing the negative cost is equivalent to minimizing the total warehouse cost.

During training, the reward is scaled before it is stored in the replay memory or used by the Actor-Critic update:

```python
scaled_reward = reward / reward_scale
```

Reward scaling improves numerical stability without changing the relative quality of different actions.

---

# Benchmark Policy

## The `(s, S)` Policy

The benchmark implements a periodic-review `(s, S)` replenishment rule.

The policy operates on the inventory position:

```text
inventory position =
    on-hand inventory
    + units already on order
```

At each daily review:

```text
if inventory position < s:
    order quantity = S - inventory position
else:
    order quantity = 0
```

Using the inventory position prevents the benchmark from repeatedly ordering units that are already in transit.

The fixed setup cost is charged only when the calculated order quantity is positive.

---

## Fair Comparison

The reinforcement learning agents and the benchmark use:

- the same demand-arrival rate;
- the same demand-size distributions;
- the same lead-time distributions;
- the same initial inventory levels;
- the same cost parameters;
- the same simulation horizon;
- the same testing seeds.

Using the same testing seeds enables paired statistical comparisons between the benchmark and the reinforcement learning agents.

---

# Double DQN

## Neural-Network Architecture

The Double DQN receives a four-dimensional state and produces one Q-value for each of the 961 combined actions.

The network architecture is:

```text
Input: 4 state variables

Fully connected layer: 64 units, ReLU
Fully connected layer: 128 units, ReLU
Fully connected layer: 64 units, ReLU

Output: 961 Q-values
```

During greedy action selection, the selected action is:

```text
argmax Q(state, action)
```

---

## Double DQN Update

The implementation uses Double DQN to reduce Q-value overestimation.

The online policy network selects the best action in the next state:

```python
next_action = argmax(policy_net(next_state))
```

The target network evaluates that action:

```python
next_q_value = target_net(next_state)[next_action]
```

The Bellman target is:

```text
target =
    reward
    + gamma * next Q-value * (1 - done)
```

If the transition terminates the episode, `done` is equal to one and the future-value term is removed.

---

## Experience Replay

Past transitions are stored in a replay memory using the following structure:

```text
(
    state,
    action,
    reward,
    next state,
    terminal flag
)
```

During training, random mini-batches are sampled from the replay memory.

Experience replay reduces the temporal correlation between consecutive transitions and allows past experiences to be reused multiple times.

---

## Epsilon-Greedy Exploration

During training:

- with probability `epsilon`, the agent selects a random action;
- otherwise, the agent selects the action with the largest predicted Q-value.

The exploration probability decreases over time according to an exponential schedule.

A nonzero final epsilon can be retained to preserve a small amount of exploration throughout training.

---

## Target Network

The target network is initialized with the same parameters as the online policy network.

Its parameters remain fixed for several episodes and are periodically replaced with the current parameters of the policy network.

This reduces instability caused by continuously changing Bellman targets.

---

## Huber Loss

The Double DQN uses Huber loss:

```python
torch.nn.functional.smooth_l1_loss
```

Huber loss behaves similarly to mean squared error for small errors and similarly to absolute error for large errors.

This makes it less sensitive to large temporal-difference errors and noisy rewards.

---

## Gradient Clipping

The gradient norm is limited before each optimizer step:

```python
torch.nn.utils.clip_grad_norm_(
    policy_net.parameters(),
    max_norm=1.0
)
```

Gradient clipping reduces the risk of unstable parameter updates.

---

## Double DQN Stabilization Techniques

The implementation includes:

- experience replay;
- a separate target network;
- Double DQN action selection and evaluation;
- epsilon-greedy exploration;
- state normalization;
- reward scaling;
- Huber loss;
- gradient clipping;
- different seeds for different training episodes;
- periodic target-network updates.

---

# Actor-Critic

## Model Structure

The Actor-Critic model contains:

1. a shared feature extractor;
2. an actor head;
3. a critic head.

### Shared Network

```text
Input: 4 state variables

Fully connected layer: 128 units, ReLU
Fully connected layer: 128 units, ReLU
Fully connected layer: 64 units, ReLU
```

### Actor Head

The actor produces one logit for each of the 961 combined actions:

```text
Fully connected layer: 64 units, ReLU
Output: 961 action logits
```

A categorical probability distribution is constructed from these logits.

During training, the action is sampled from this distribution.

During deterministic evaluation, the action with the largest logit is selected.

### Critic Head

The critic estimates the value of the current state:

```text
Fully connected layer: 64 units, ReLU
Output: 1 state value
```

---

## Discounted Returns

The discounted return is calculated as:

```text
G_t = reward_t + gamma * G_(t+1)
```

The rewards are processed backward from the final step of the episode.

The returns are normalized before the parameter update to improve numerical stability.

---

## Advantage Estimate

The advantage is estimated as:

```text
advantage =
    discounted return
    - estimated state value
```

A positive advantage means that the selected action performed better than the critic expected.

A negative advantage means that it performed worse than expected.

---

## Actor Loss

The actor loss is:

```text
actor loss =
    -mean(log probability * advantage)
```

This objective increases the probability of actions associated with positive advantages and decreases the probability of actions associated with negative advantages.

---

## Critic Loss

The critic is trained to approximate the discounted return.

The implementation uses Huber loss:

```python
critic_loss = smooth_l1_loss(
    estimated_values,
    discounted_returns
)
```

---

## Entropy Bonus

An entropy term is added to encourage exploration.

The complete objective is:

```text
total loss =
    actor loss
    + value coefficient * critic loss
    - entropy coefficient * entropy
```

Higher entropy corresponds to a less deterministic action distribution.

The entropy bonus discourages the actor from collapsing too early to a single action.

---

## Actor-Critic Gradient Clipping

Before the optimizer step, the gradient norm is limited to:

```python
max_norm = 1.0
```

This reduces the risk of unstable updates.

---

# Training Procedure

## Training Seeds

A different seed is used for every training episode:

```python
episode_seed = initial_seed + episode_number
```

This exposes the agent to different:

- demand-arrival sequences;
- demand quantities;
- lead-time realizations;
- inventory trajectories.

Using multiple training seeds reduces the risk of overfitting to a single stochastic trajectory.

---

## Evaluation Seeds

A fixed collection of seeds is used during evaluation.

The same evaluation seeds are used for:

- Double DQN;
- Actor-Critic;
- the `(s, S)` benchmark.

This supports a paired comparison between policies.

---

# Evaluation Metrics

The following metrics are collected for each replication:

- average daily total cost;
- cumulative holding cost;
- cumulative shortage cost;
- cumulative ordering cost;
- total episode reward;
- average order quantity for each product;
- maximum selected order quantity;
- percentage of days with no order.

The component-level metrics help identify whether poor performance is caused by:

- excessive inventory;
- frequent or large orders;
- insufficient replenishment;
- delayed replenishment decisions.

---

## Interpretation of the Comparisons

The reported difference is calculated as:

```text
S-min/S-max cost - reinforcement learning cost
```

Therefore:

- a positive value means that the reinforcement learning agent has a lower cost;
- a negative value means that the `(s, S)` benchmark has a lower cost;
- a confidence interval containing zero indicates that the observed difference is not statistically conclusive at the selected confidence level.

---

# Statistical Analysis

## Paired Comparisons

Because every policy is evaluated with the same testing seeds, the results are compared using paired differences.

For replication `r`:

```text
difference_r =
    benchmark cost_r
    - agent cost_r
```

The sample mean and confidence interval of the paired differences are then calculated.

---

## Confidence Intervals

Two-sided Student's t confidence intervals are used.

For significance level:

```text
alpha = 0.05
```

the critical value is calculated as:

```python
stats.t.ppf(
    1 - alpha / 2,
    n - 1
)
```

The confidence interval is:

```text
sample mean +/- half interval
```

---

## Welch Warm-Up Analysis

Welch's method is used to estimate an initial transient period from the average daily-cost process of the benchmark policy.

The procedure:

1. calculates the average process across simulation replications;
2. smooths the average process using a moving average;
3. evaluates changes in the smoothed process;
4. identifies a potential point after which the process is approximately stable.

The estimated warm-up period is excluded when calculating the benchmark average daily cost.

---

# Running the Notebook

Execute the notebook cells in the following order.

## 1. Shared Definitions

Run Cell 1 to define:

- global parameters;
- the simulation models;
- the warehouse environment;
- Double DQN;
- the `(s, S)` benchmark;
- statistical utilities.

## 2. Double DQN Experiment

Run Cell 2 to:

- train the Double DQN;
- evaluate the trained policy;
- evaluate the benchmark;
- calculate confidence intervals;
- compare cost components;
- analyze selected actions;
- generate plots.

## 3. Actor-Critic Experiment

Run Cell 3 to:

- train the Actor-Critic model;
- evaluate it on the same test seeds;
- evaluate the benchmark;
- compare cost components;
- analyze selected actions;
- generate plots.

Training should be repeated if the state representation, maximum order quantity, neural-network architecture, or action space is changed.

---

# Configuring the Maximum Order Quantity

The maximum order quantity is controlled by:

```python
MAX_ORDER_QTY = 30
```

The action-space dimensions are calculated automatically:

```python
ACTION_SIZE = MAX_ORDER_QTY + 1
TOTAL_ACTIONS = ACTION_SIZE * ACTION_SIZE
```

For example, if:

```python
MAX_ORDER_QTY = 20
```

then:

```text
ACTION_SIZE = 21
TOTAL_ACTIONS = 441
```

If:

```python
MAX_ORDER_QTY = 30
```

then:

```text
ACTION_SIZE = 31
TOTAL_ACTIONS = 961
```

Both reinforcement learning models must be retrained after changing `MAX_ORDER_QTY`.

---

# Dependencies

The notebook uses:

- Python;
- NumPy;
- Pandas;
- Matplotlib;
- SciPy;
- SimPy;
- PyTorch;
- Gymnasium;
- tqdm.

The main external dependencies can be installed with:

```bash
pip install gymnasium simpy
```

A CUDA-compatible GPU is not required.

---

# Reproducibility

The implementation controls the random generators used by:

```python
random
numpy
torch
```

Training episodes use different seeds, while evaluation uses a fixed list of seeds.

Small numerical differences can still occur across hardware platforms, Python versions, or library versions.

---

# Results

The final results should be added after completing the selected training runs.

## Double DQN vs `(s, S)`

| Metric | `(s, S)` | Double DQN | Difference `(s, S) - DQN` |
|---|---:|---:|---:|
| Average daily cost | To be added | To be added | To be added |
| Holding cost | To be added | To be added | To be added |
| Shortage cost | To be added | To be added | To be added |
| Ordering cost | To be added | To be added | To be added |

## Actor-Critic vs `(s, S)`

| Metric | `(s, S)` | Actor-Critic | Difference `(s, S) - Actor-Critic` |
|---|---:|---:|---:|
| Average daily cost | To be added | To be added | To be added |
| Holding cost | To be added | To be added | To be added |
| Shortage cost | To be added | To be added | To be added |
| Ordering cost | To be added | To be added | To be added |

---

# Limitations

The current implementation has several limitations:

1. unmet demand is represented as backlog rather than lost sales;
2. the two product decisions are encoded as one combined action;
3. the action space grows quadratically with the maximum order quantity;
4. the Actor-Critic model is updated once per complete episode;
5. model hyperparameters are selected manually;
6. the benchmark parameters are fixed rather than optimized;
7. demand distributions are assumed to remain stationary;
8. warehouse capacity is not constrained;
9. supplier capacity is not constrained;
10. the model does not include order cancellation or emergency replenishment.

---

# Possible Improvements

Potential future extensions include:

- Prioritized Experience Replay;
- Dueling Double DQN;
- soft target-network updates;
- separate action heads for the two products;
- Proximal Policy Optimization;
- automatic hyperparameter optimization;
- optimized `(s, S)` benchmark parameters;
- supplier-capacity constraints;
- storage-capacity constraints;
- nonstationary demand;
- seasonal demand;
- service-level metrics;
- lost-sales modeling;
- additional products;
- parallel simulation environments;
- longer evaluation experiments;
- separate random-number streams for demand and lead times.

---

# Conclusion

This project formulates a two-product warehouse replenishment problem as a reinforcement learning task.

A common discrete-event simulation is used to evaluate:

- a traditional `(s, S)` replenishment policy;
- a Double DQN agent;
- an Actor-Critic agent.

The agents observe both the on-hand inventories and the outstanding-order quantities for the two products.

The final comparison considers the total cost and its individual components. This makes it possible to determine whether the performance of a policy is mainly affected by excessive inventory, frequent ordering, or insufficient replenishment leading to shortages.
