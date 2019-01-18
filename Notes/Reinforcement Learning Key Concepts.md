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

3. **Optimal Value Function**, $V^{*}(s)$ gives the expected return if you start in state $s$, and always act according to the *optimal* policy $\pi$:

   ​	$Q^{\pi}(s,a) = max_{\pi}\space \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s, a_{0} = a]​$	

4. **Optimal Action-Value Function**, $Q^{*}(s,a) = max_{\pi}\space \Epsilon_{\tau \sim \pi} [R(\tau) | s_{0} = s, a_{0} = a]​$

