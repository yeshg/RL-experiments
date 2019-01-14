# Soft Actor Critic (SAC)

## Issue with typical Model-free Deep RL algorithms

- very high sample complexity
- brittle convergence patterns leading to lots of tough hyperparameter tuning

These issues limit the applicability of these methods to complex real-world problems.



## Overview

Soft-Actor-Critic is an off-policy actor-critic deep RL aglorithm based on maximum entropy reinforcement learning framework. Basic idea is that the actor wants to maximize the expected reward while also maximizing entropy (wants to suceed at task while acting as random as possible, so exploring the environment).

Experiments show that SAC outperforms prior on-policy and off-policy methods.

### Difference between On-policy and Off-policy

#### Off Policy:

Q-learning is off-policy because it updates its Q-values using the Q-value of the next state $s'$ and the greedy action $a'$. It estimates the return (total discounted future reward) for state-action pairs assuming a greedy policy was followed, even though it's not actually following a greedy policy.

#### On Policy:

Something like Vanilla Policy Gradient methods is on policy because it updates its Q-values using the Q-value of the next state $s'$ and the current policy's action $a''​$. It estimates the return for state-action pairs assuming the current policy continues to be followed.

If the current policy is a greedy policy, the on-policy becomes no different from the off-policy methods. This would be bad though, because it would never explore.

### Why On-Policy Learning causes poor sample efficiency

Policy-based methods like TRPO, PPO, and A3C all require new samples to be collected for each gradient step. The number of gradient steps and samples for sptep needed to learn an effective policy increases with task complexity. Off-policy methods aim to reuse past experience.

### Combining a model-free deep RL algorithm with the maximum entropy RL framework

Maximum entropy RL alters the RL objective to include exploration as well as high return.

3 components:

1. actor-critic architecture with separate policy and value function networks
2. off-policy formulation that enables the reuse of previously collected data for efifciency
3. entropy maximization for stability and exploration

### DDPG

DDPG is a popular off-policy actor-critic method is a deep variant of the deterministic policy gradient algorithm. It uses a Q-function estimator to enable off-policy learning and a deterministic actor that maximizes the Q-function. This makes DDPG a hybrid between Q-learning algorithm and deterministic actor-critic.

## Preliminaries

### Notation

Infinite Horizon Markov decision process (MDP) defined by the tuple $(S,A,p,r)$, where the state space $S$ and the action space $A$ are continuous and the unknown state transition probability $p : S \times S \times A \to  [0,\infty)$ represents the probability of the next state $s_{t+1}$ given the current state $s_{t}$ and the action $a_{t}$.

$p_{\pi}(s_{t})$ and $p_{\pi}(s_{t},a_{t})$ to denote the state and state-action marginals of the trajectory distribution induced by the policy noted by $\pi(a_{t},s_{t})$. 

### Maximum Entropy Reinforcement Learning

Standard RL only maximizes the expected sum of rewards:

$\sum_{t} E_{(s_{t},a_{t}) \sim p_{\pi}}[r(s_{t},a_{t})]​$

Instead, we will maximize an more general entropy objective that favors stochastic policies by augmenting the aforementioned objective function with the exprected entropy of the poilcity over $p_{\pi}(s_{t})​$:

$J(\pi) = \sum\limits_{t=0}^TE_{(s_{t},a_{t}) \sim p_{\pi}}[r(s_{t},a_{t}) + \alpha \cal H(\pi(\cdot|s_{t}))] $

$\alpha$ is the "temperature parameter". It determines the relative importance of the entropy term against the reward. It controls how randomly determined the optimal policy is. To get the conventional reward objective for RL, take the limit of the above function as $\alpha \to 0$.

This objective hopes to cause following:

- policy should be incentivized to explore more widely while giving up on the blatantly unpromising possibilities
- should be able to capture multople modes of near optimal behavior

## Entropy-Regularized Reinforcement Learning

if $x$ is a random variable with a probability density $p$, its entropy $\cal H$ is computed from its distribution $p$:

$\cal H(P) = E_{x \sim P}[-log  P(x)]$.

In entropy regularized RL, the agent is given a bonus reward at each time step that is proportional to the entropy of ht epolicy at that timestep (this incentivizing the policy itself to have a high entropy at that timestep). This changes the RL problem to:

$\pi^{*} = arg max_{\pi} E_{T \sim \pi}[\sum \limits_{t=0}^{\infty}\gamma^{t}(R(s_{t},a_{t},s_{t+1}) + \alpha \cal H(\pi(\cdot|s_{t})))]​$

This causes the value function to also be slightly different as we need to include the entropy bonuses from every timestep:

$V^{\pi}(s) = E_{T \sim \pi} [\sum \limits_{t=0}^{\infty}\gamma^{t}(R(s_{t},a_{t},s_{t+1}) + \alpha \cal H(\pi(\cdot|s_{t}))) \mid s_{0} = 0]$

The Q-function $Q^{\pi}$ is changed to include the entropy bonuses from every timestep except the first:

