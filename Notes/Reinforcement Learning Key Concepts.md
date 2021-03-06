# Reinforcement Learning Key Concepts

based on https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

[TOC]

RL is centered around training **agents** by trial and error. It does this by rewarding or punishing an agent based on that agent's behavior, which causes the agent's future behavior to change.

## Barebones RL: The Agent and the Environment

The **agent** interacts with the **environment**. At every **step**, the agent sees an observation (could be partial observation) of the state of the world, and decides on an action to take. The environment can change both on its own and when acted upon by the agent.

The agent will get a **reward** signal from the environment - a number that indicates how good or bad the current world state is. The agent's goal is to maximize cumulative future reward aka **return**. There are a bunch of reinforcement learning algorithms/methods to help agents maximize their return.

## States and Observations

A **state** $s$ is a complete description of the world. An **observation** $o$ is a partial description of a state which may or may not be complete.

States and observations are generally represented by a *real-valued vector, matrix, or higher-order tensor* (an n-dimensional table of real numbers). For example, a visual observation is represented by an RGB matrix (4-D). The state of a robot could be represented by joint angles and velocities.

When an agent is able to observe the complete state of the environment, that environment is considered **fully observed**. Likewise, if an agent only sees partial observations, the environment is **partially-observed**.

Note: Reinforcement learning literature often interchanges the $s$ for state with the $o$ for observation. This notation is used when the literature discuesses how an agent decides an action (based on the state $s$ or observation $o$). This is important because research papers may indicate that the action choice is conditioned on the state when in practice it's actually conditioned on an incomplete observation.

## Action Spaces

The **action space** is the set of all valid actions in a given environment. Some environments have **discrete action spaces** while others have **continuous action spaces**. In the case of robotics, a continuous action space is used. In continuous spaces, actions are represented by real-valued vectors (just like states).

Many RL algorithms can only be applied directly to one type of action space, or would need a lot of rework for being applied to the other, so the distinction is important.

## Policies

Policies are the brains of RL agents: they essentially dictate what actions the agent should take. A **policy** is a rule that is conditioned on the state (it could also be conditioned on observation, but from here on I will only refer to the state).

A **deterministic** policy is usually denoted by $\mu$:

$a_{t} = \mu(s_{t})​$

A **stochastic** policy is usually denoted by $\pi$.

$a_{t} \sim \pi(\cdot|s_{t})​$

In literature, "policy" is often used to replace "agent" because the policy is actually what decides what actions to take, and is therefore trying to maximize reward.

Deep RL uses **parameterized policies**: policies that are have the interface of a normal function with the internal workings of a neural network. This network's parameters are adjusted during training via some optimization algorithm to make the policy maximize return.

The parameters of a policy are usually denoted by $\theta$ or $\phi$ and are written as subscripts on the policy function symbol:

$a_{t} = \mu_{\theta}(s_{t})$

$a_{t} \sim \pi_{\theta}(\cdot|s_{t})$

### Deterministic Policies

Example below is a deterministic policy for a continuous action space in TensorFlow:

```python
obs = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
net = mlp(obs, hidden_dims=(64,64), activation=tf.tanh)
actions = tf.layers.dense(net, units=act_dim, activation=None)
```

Here, `mlp` is a function that stacks multiple `dense` (TensorFlow's word for FC layer) on top of each other with the given sizes and the tanh activation function.

### Stochastic Policies

There are two main steps in stochastic policies:

- Sampling actions from the policy
- computing the log likelihoods of particular actions, $log(\pi_{\theta}(a|s))$

There are two kinds of stochastic policies on RL: **categorical policies** (for discrete action spaces) and **diagonal Gaussian policies** (for continuous action spaces).

#### Categorical Policies

Categorical policies are used in discrete action spaces, while diagonal Gaussian policies are used in continuous action spaces.

A categorical policy is like a classifier over discrete actions. The neural network is built the same way as a classifier: the input is the observation, the input is processed by layers of the neural network, the neural network concludes with a final linear layer to give the logits for each action which is then passed through a softmax to convert the logits into probabilities.

- **Sampling**: given the probabilities for each action, you can sample with built-in-tools for ML frameworks. TensorFlow has tf.distributions.Categorical and pyTorch has torch.multinomial()...
- **Log-Likelihood**: The log-likelihood for an action $a$ is simply found by taking the log of a particular indexes value in the column vector outputted by the neural network: $log\space \pi_{\theta}(a|s) = log[P_{\theta}(s)]_{a}$, where $P_{\theta}(s)$ is the last layer of probabilities

#### Diagonal Gaussian Policies

A *multivariate* Gaussian distribution is described by a mean vector, $\mu$, and a covariance matrix, $\Sigma$. A *diagonal* Gaussian distribution is a special case of this where the covariance only has entries on the diagonal. This lets us reduce the covariance matrix to a single vector.

A diagonal Gaussian policy always has a neural network that maps from observations to mean actions, $\mu_{\theta}(s)$. There are two different ways that the covariance matrix (single vector) is typically represented:

- There is a single vector of log standard deviations, $log \space \sigma$, which is not a function of state.
- There is a neural network that maps from states to log standard deviations, $log \space \sigma_{\theta} (s)$. It can optionally share some layers with the mean network

Log standard deviations are used instead of just standard deviations because log stds can range from $(-\infty,\infty)​$ while stds must be positive. This makes training easier. To go back to standard deviations, simply exponentiate the logs.

For Diagonal Gaussian Policies, sampling is done by obtaining a vector of noise from a spherical Gaussian ($z \sim \cal N(0,I)$). Then, an action sample is computed with 

$a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$

Log-likelihood: The log-likelihood of a $k$-dimensional action $a$, for a diagonal Gaussianw ith mean $\mu = \mu_{\theta}(s)$ and standard deviation $\sigma = \sigma_{\theta}(s)$.

$log \space \pi_{\theta} (a|s) = -\frac{1}{2}(\sum\limits_{i=1}^{k}(\frac{(a_{i}-\mu_{i})^{2}}{\sigma_{i}^{2}}+2log\space\sigma_{i})+k\space log\space2\pi)$.

## Trajectories

A **trajectory** $\tau$ is a sequence of states and actions in the world,

$\tau = (s_{0},a_{0},s_{1},a_{1},...).​$ 

The very first state of the world, $s_{0}$, is randomly sampled from the **start-state distribution**, which is denogted by $\rho_{o}(\cdot)$.

State transitions are what happens to the world between one timestep and the next are governed by the natural laws of the environment, and depend on only the most recent action, $a_t​$.

They can be deterministic: $s_{t+1} = f(s_{t},a_{t})$

Or stochastic: $s_{t+1} \sim P(\cdot | s_{t},a_{t})$.

Other words for trajectories include **episodes** or **rollouts**.

##Reward and Return

The reward function $R​$ depends on the current state of the world, the action just taken, and the next state of the world:
$$r_{t} = R(s_{t}, a_{t}, s_{t+1})​$$

The agent's goal is to maximize cumulative reward over a trajectory, but there are different definitions of this.

One kind of return is the **finite-horizon undiscounted return**, which is the sum of rewards obtained in a fixed window of steps:
$R(\tau) = \sum\limits_{t=0}^{T}r_{t}​$.

Another kind is the **infinite-horizon discounted return**, which is the sum of all rewards ever obtained by the agent, discounted by how far off in the future they're obtained. The discount factor is important for the exploration (to hopefully exploit higher future returns) vs exploitation tradeoff. Anyways, an infinite-horizon sum of rewards may not converge to a finite value but with a discount factor it is ensured to:

$R(\tau) = \sum\limits_{t=0}^{\infty}\gamma^{t} r_{t}$, where $\gamma \space\epsilon\space (0,1)$

##The RL Problem

Regardless of the choice of policy or return measure, the goal of all RL is to select a policy that maximizes **expected return** when the agent acts according to it.

If the environment transitions and the policy are stochastic, the probability of a $T$-step trajectory is:

$P(\tau | \pi) = \rho_{0}(s_{0})\prod \limits_{t=0}^{T-1}P(s_{t+1}|s_{t},a_{t})\pi(a_{t}|s_{t})$, recall that $\rho_{0}$ is the start-state distribution.

The expected return denoted by $J(\pi)$ is:

$J(\pi) = \int_{\tau}P(\tau|\pi)R(\tau) = \Epsilon_{\tau \sim \pi}[R(\tau)]$.

The central optimization problem in RL is then expressed by:

$\pi^{*} = arg \space max_{\pi} \space J(\pi)$, where $\pi^{*}$ is the optimal policy.

## Value Functions

Although they aren't actually necessary for a barebones RL problem, it's often useful to know the **value**, or expected return, of a state or state-action pair. Value functions are used in almost every RL algorithm.

There are four main functions of note:

1. **On-Policy Value Function**, $V^{\pi}(s)$, gives the expected return if you start in state $s$ and always act according to policy $\pi$:

   ​	$V^{\pi}(s) = \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s]​$

2. **On-Policy Action-Value Function**, aka the On-Policy Q-function, $Q^{\pi}(s,a)​$ gives the expected return if you start in state $s​$, take some action $a​$ that may or may not have come from the policy, and then forever after act according to the policy $\pi​$:

   ​	$Q^{\pi}(s,a) = \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s, a_{0} = a]$

3. **Optimal Value Function**, $V^{*}(s)​$ gives the expected return if you start in state $s​$, and always act according to the *optimal* policy $\pi​$:

   ​	$V^{*}(s) = max_{\pi}\space \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s]$	

4. **Optimal Action-Value Function**, $Q^{*}(s,a) = max_{\pi}\space \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s, a_{0} = a]​$, gives the expected return if you start in a state $s​$, take an arbitrary action $a​$, and then forever after act according to the optimal policy in the environment.

Note that the above functions don't depend on a time argument. This works only for infinite-horizon disocunted return... if we wanted to calculate finite-horizon undiscounted return we would need a value function that accepts time as an argument.

### Connection between value function and action-value (Q) function

$V^{\pi}(s) = \Epsilon _{a \sim \pi}[Q^{\pi}(s,a)]$

$V^{*}(s) = \max _{a} \Epsilon _{a \sim \pi}[Q^{*}(s,a)]$

## Optimal Q-Function and the Optimal Action

There is an important connection between optimal action-value function $Q^{*}(s,a)$ and the action selected by the optimal policy. By definition, $Q^{*}(s,a) $ gives the expected return for starting in state $s$, taking an arbitrary action $a$, and then acting according to the optimal policy forever after.

The optimal policy in $s$ will select whichever action maximizes the expected return from starting in $s$:

$a^{*}(s) = arg \space max_{a} Q^{*}(s,a)$.

## Bellman Equations

All four of the value function variants obey special self-consistency equations called **Bellman Equations**. The basic idea behind the bellman equations is *<u>the value of your starting point is the reward you expect to get from being there plus the value of wherever you land next</u>*.

Bellman equations for on-policy value functions are

$V^{\pi}(s) = \Epsilon_{a \sim \pi,s'\sim P}[r(s,a)+\gamma V^{\pi}(s')])​$,

$Q^{\pi}(s,a) = \Epsilon_{s' \sim P}[r(s,a)+\gamma \Epsilon_{a' \sim \pi}[Q^{\pi}(s',a')]]​$.

Here, $s' \sim P$ is short for $s' \sim P(\cdot|s, a)$. This means that the next state $s'$ is sampled from the environment's transition rules; $a \sim \pi$ is shorthand for $s' \sim \pi(\cdot | s)$; and $a' \sim \pi$ is short for $a' \sim \pi(\cdot | s')$.

Bellman equations for optimal value functions are similar, they just include the max operator (choosing the action that is optimal):

$V^{*}(s) = max_{a} \space\Epsilon_{a \sim \pi}[r(s,a)+\gamma V^{*}(s')])$,

$Q^{*}(s,a) = \Epsilon_{s' \sim P}[r(s,a)+\gamma \space max_{a'} \space Q^{*}(s',a')]$.

**Bellman backup** refers to the right-hand side of the bellman equation (reward + next value).

## Advantage Functions

**Advantage functions** are used to describe how good an action is on a relative sense. The advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot | s)$, assuming you act according to the policy $\pi$ forever after. Mathematically it is described as follows:

$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$.

The advantage function is crucial for policy gradient methods.

## Formalizing Rl into MDP

A Markov Decision Process (MDP) is a 5-tuple, $<S,A,R,P,\rho_{0}>$ where

- $S​$ is the set of all valid states
- $A$ Is the set of all valid actions
- $R : S \times A \times S \to \mathbb{R}​$ is the reward function, with $r_{t} = R(s_{t}, a_{t}, s_{t+1})​$
- $P : S \times A \to \cal P$ is the transition probability function, with $P(s'|s,a)$ being the probability of transitioning into state $s'$ if you start in state $s$ and take action $a$.
- $\rho_{0}$ is the starting state distribution.

The name Markov Decision Process refers to the Markov property, which states that transitions only depend on the most recent state and action, and no prior history.

# Types of RL Algorithms

from: https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html

![](/Users/yeshg/RL/RL-experiments/Notes/img/rl_algorithms_9_15.svg)

## Model-Free vs Model-Based RL

Question over whether or not the agent has access to (or learns) a **model** of the environment. A model of the environment is a function that predicts state transitions and rewards.

The main upside to Model-based RL is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. The agents can then distill the results from planning ahead into a learned policy. AlphaZero (the RL algo that became superhuman at Chess and Shogi) is a famous example of a model-based approach. In general, model-based approaches have a substantial improvement in sample efficiency over methods that don't have a model.

The main downside to Model-based approaches is that a **ground-truth model of the environment is usually not available** to the agent, so an agent that wants to use a model would need to learn a model from experience. This creates several challenges, the biggest one being that the bias in the model can be exploited by the agent, resulting in an agent that performs well with respect to the learned model but behaves sub-optimally for the real environment (kinda like overfitting). Model-learning is hard, so efforts to learn it can fail to pay off.

Algorithms which use a model are called **model-based**, those that don't are **model-free**. Model-free methods give up the potential gains in sample efficiency from using a model but tend to be easier to implement and tune.

## What to Learn

There are several things that RL algorithms can learn:

- Policies (can be stochastic or deterministic)
- Action-value functions (aka Q-functions)
- value functions
- And/or environment models (model-based RL)

### Learning in Model-Free RL

There are two main approaches to representing and training agents with model-free RL, as well as a third hybrid approach.

#### Policy Optimization

Methods in this approach represent a policy as $\pi_{\theta}(a|s)$, where $\theta$ are the parameters. $\theta$ can be optimized directly either by gradient ascent on the performance objective $J(\pi_{\theta})$, or indirectly by maximizing local approximations of $J(\pi_{\theta})$. This optimization is **performed <u>on-policy</u>, meaning that each update only uses data collected while acting according to the most recent version of the policy**. Policy optimization usually involves learning an approximation $V_{\phi}(s)$ for the on-policy value function $V^{\pi}(s)$, which is used for figuring out how to update the policy.

Some examples of policy optimization methods are:

- [A2C](https://arxiv.org/abs/1602.01783) / [A3C](https://arxiv.org/abs/1602.01783) - perform gradient ascent to directly maximize the performance objective
- [PPO](https://arxiv.org/abs/1707.06347) - updates indrectly maximize performance by instead maximizing a *surrogate objective* function that gives a safe estimate for how much $J(\pi_{\theta})$ will change as a result of the update (so as to not destroy the policy)

#### Q-Learning

Q-functions are Action-value functions. Q-learning methods learn an approximation $Q_{\theta}(s,a)$ for the optimal action-value function $Q^{*}(s,a)$. They typically use a function based on the Bellman Equation (basic idea of bellman equation is "the value of your starting point is the reward you expect to get from being there plus the value of wherever you land next"). This optimization is almost always performed **off-policy, meaning that each update can use data collected at any point during training** (leading to the idea of experience replay buffers), regardless of how the agent was choosing to explore the environment when the data was obtained. The corresponding policy is obtained via the connection between the optimal policy function $\pi^{*}$ and the optimal Action-Value function $Q^{\theta}$:

The actions taken by the Q-learning agent are given by $a(s) = arg \space max_{a} \space Q_{\theta}(s,a)$.

Some examples of Q-learning methods include:

- [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), the deep-q-learning algo that launched Deep RL
- [C51](https://arxiv.org/abs/1707.06887), a variant that learns a distribution over return whose expectation is $Q^{*}$

#### Trade-offs between Policy Optimization and Q-Learning

Primary strength of policy optimization methods is that they are principled - *you directly optimize for the thing you want*. This makes them stable and reliant. Q-Learning methods only *indirectly* optimize for agent (because they train $Q_{\theta}$ to satisfy a self-consistency (Bellman) equation). Q-learning methods are not as stable, but are substantially more sample efficient when they do work because they reuse data more effectively than policy optimization techniques.

#### Interpolating Between Policy Optimization and Q-Learning

Surprisingly (and a good thing), policy optimization and Q-learning are not incompatible (and can be [equivalent](https://arxiv.org/abs/1704.06440)). There are several algorithms that lie between them, and try to trade-off the strengths and weaknesses between them.

Examples:

- [DDPG](https://arxiv.org/abs/1509.02971), an algorithm which concurrently learns a deterministic policy and a Q-function by using each to improve the other
- [SAC](https://arxiv.org/abs/1801.01290), a variant which uses stochastic policies, entropy regularization, and a few other tricks to stabilize learning and score higher than DDPG on standard benchmarks.

### Learning in Model-Based RL

There aren't a small number of easy-to-define clusters of methods for model-based RL: there are many independent ways of using models. A few examples below.

#### Background: Pure-Planning:

The most basic approach *never* explicitly represents the policy. Instead it uses pure planning techniques like model-predictive control ([MPC](https://en.wikipedia.org/wiki/Model_predictive_control)) to select actions. In MPC, each time the agent observes the environment, it computes a plan which is optimal with respect to the model, where the *plan describes all actions to take over some fixed window of time after the present* (Future reward beyond the horizon may be considered by the planning algorithm through the use of a learned value function). The agent then executes the first action of the plan, and immediately discards the rest of it. It computes a new plan each time it prepares to interact with the environment to avoid using an action froma a plan with a shorter-than-desired planning horizon.

- The [MBMF](https://sites.google.com/view/mbmf) work explores MPC with learned environment models on some standard benchmark tasks for deep RL.

#### Expert Iteration

An improvement upon pure-planning involves using and learning an explicit representation of the policy, $\pi_{\theta}(a|s)$. The agent uses a planning algorithm (like Monte Carlo Tree Search) in the model and generates candidate actions for the plan by sampling from its current policy. The planning algorithm produces an action which is better than what the policy alone would have produced (hence being called an "expert" relative to the policy). Then, the policy is updated to produce an action more like the planning algorithm's output.

- The [Exlt](https://arxiv.org/abs/1705.08439) algorithm uses this approach to train deep neural networks to play Hex
- [AlphaZero](https://arxiv.org/abs/1712.01815) also falls under this approach

#### Data Augmentation for Model-Free Methods (*imagination*)

In these approaches, you use a model-free RL algorithm to train a policy or Q-function, but either

1. Augment real experiences with ficitcious ones in updating the agent
2. OR only use fictitious experiences for updating the agent

Examples:

- [MBVE](https://arxiv.org/abs/1803.00101) - augmenting real experiences with ficticious ones
- [World Models](https://worldmodels.github.io) - using purely ficticious experience to train the agent, which they call "training in the dream"

#### Embedded Planning and Loops into Policies

Another approach treats the planning procedure as a subroutine of the policy - making complete plans side information for the policy which is trained with any standard model-free algorithm. The key concept is that in this framework, the policy can learn to choose how and when to use the plans. This makes model bia less of a problem, because if the model is bad for planning in some states the policy can choose to ignore it.

- [I2As](https://arxiv.org/abs/1707.06203) is an architecture for RL that shows several examples of this approach.

https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html

https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html