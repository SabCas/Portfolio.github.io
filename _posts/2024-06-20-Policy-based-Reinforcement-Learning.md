## Introduction to REINFORCE

REINFORCE is a Monte Carlo policy gradient algorithm used to optimize a policy in reinforcement learning. The main idea is to adjust the parameters of a policy network to maximize the expected cumulative reward. This involves sampling trajectories from the environment, computing the returns, and using these to update the policy parameters.

## Mathematical Foundation

### Policy Gradient Theorem

Given a policy $( \pi_\theta(a|s) )$ parameterized by $( \theta )$, the objective is to maximize the expected return $( J(\theta) )$:

$$J(\theta) = \mathbb{E}_\pi [ G_t ]$$

where $( G_t )% is the return from time step $( t )$.

In reinforcement learning, the return $( G_t )$ is defined as the total accumulated reward from time step $( t )$ onwards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

where $( \gamma $) is the discount factor and $( r_{t+k} )$ is the reward received at time step $( t+k )$.

### Deriving the Policy Gradient

The policy gradient theorem states that the gradient of the objective function can be expressed as:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi [ \nabla_\theta \log \pi_\theta(a|s) G_t ]$$

To understand why this is true, let's derive it step-by-step.

#### Step 1: Express the Objective Function

First, we express the objective function \( J(\theta) \) in terms of the policy \( \pi_\theta \) and the state-action value function \( Q^\pi(s, a) \):

$$J(\theta) = \sum_s d^\pi(s) \sum_a \pi_\theta(a|s) Q^\pi(s, a)$$

where $( d^\pi(s) )$ is the stationary distribution of states under policy $( \pi_\theta )$.

#### Step 2: Differentiate the Objective Function

Next, we differentiate $( J(\theta) )$ with respect to $( \theta )$:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_s d^\pi(s) \sum_a \pi_\theta(a|s) Q^\pi(s, a)$$

Using the linearity of the gradient operator:

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a|s) Q^\pi(s, a)$$

#### Step 3: Apply the Log-Derivative Trick

We apply the log-derivative trick $( \nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) )$:

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)$$

Since \( \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a) \) is an expectation over actions, we can rewrite it as:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a) ]$$

#### Step 4: Use the Return as an Estimate

In practice, we do not have access to the true state-action value function $( Q^\pi(s, a) )$, so we use the sampled return $( G_t )$ as an unbiased estimate of $( Q^\pi(s, a) )$:

$$\nabla_\theta J(\theta) \approx \mathbb{E}_\pi [ \nabla_\theta \log \pi_\theta(a|s) G_t ]$$

This is the policy gradient theorem, which forms the basis for the REINFORCE algorithm.

### REINFORCE Update Rule

Using Monte Carlo methods, we can estimate the gradient using sampled trajectories. The update rule for the policy parameters \( \theta \) is:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

where $( \alpha )$ is the learning rate. Substituting the gradient, we get:

$$\theta \leftarrow \theta + \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t$$

### PyTorch Implementation

Let's implement the REINFORCE algorithm in PyTorch step-by-step.

#### Step 1: Define the Policy Network

The policy network takes the current state as input and outputs a probability distribution over actions.

---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
```

### Step 2: Define the Agent
The agent uses the policy network to interact with the environment, store rewards and actions, and update the policy.
---
```
class Agent:
    def __init__(self, state_space, action_space, lr, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.policy = PolicyNetwork(state_space, action_space, lr).to(device)
        self.reward_memory = []
        self.action_memory = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        probabilities = self.policy.forward(state)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.action_memory.append(log_prob)
        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = torch.tensor(G, dtype=torch.float).to(device)
        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob
        loss.backward()
        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []
```

### Step 3: Training the Agent
We can now train the agent by letting it interact with the environment and updating the policy based on the rewards it receives.

---
```
import gym

env = gym.make('LunarLander-v3')
agent = Agent(state_space=env.observation_space.shape[0], action_space=env.action_space.n, lr=0.01)

n_episodes = 1000

for episode in range(n_episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.store_reward(reward)
        state = state_
        score += reward
    agent.learn()
    print(f'Episode {episode}, Score: {score}')
```
