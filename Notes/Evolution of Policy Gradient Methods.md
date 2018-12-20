# Evolution of Policy Gradient Methods

## Vanilla Policy Gradient Method (covers Actor-Critic)

### Understanding The Policy Gradient Loss Function (used in vanilla pg methods and Actor-Critic:

$\hat E_{t}[log\pi_{\theta}(a_{t}|s_{t})\hat A_{t}]$

In English, this is "the expectation over the log of the policy actions times an estimate of the advantage function"

The first term $\pi_{\theta}(a_{t}|s_{t})$ is the policy ( our NN that takes in state and outputs an action distribution). We use log to make this probability distribution differentiable.

Second term $\hat A_{t}$ is the advantage function. This is the estimate of the relative value of the selected action. It is calculated as follows:

$\hat A_{t} = $ Discounted rewards (return) - baselines estimate (value function)

Discounted rewards = $$\sum_{k=0}^{\infty }\gamma^{k}R_{t+k+1}\;where\;\gamma\;\epsilon\;[0,1)$$

This is a weighted sum of all the rewards the agent got during each timestep in the current episode.

Gamma is high means longer term reward is more important. Gamma is low means shorter term reward is more important.

The Baseline or value function gives an estimate of the (discounted) sum of rewards from the current point onwards. The value function is the neural network.

This portion is actually a supervised learning problem: The value function NN takes in states and returns estimates of the discoutned return form this point onwards. Because this estimate is coming from a neural network, it will be a noisy (has error/loss) estimate. Basically, there's going to be some variance.

Together, the difference between the disocunted rewards and the baseline estimate gives us the difference between 

1. What we KNOW happened
2. What we EXPECTED to happen

This is the **advantage estimate**, which answers the question "How much better was the action that I took based on what would normally happen in the state I was in"

Final "Vanilla" Policy Gradient Algorithm:

```
Initialize policy parameter theta, baseline b
for iteration 1,2,... do
	Collect a set of trajectories by executing the current policy
	At each timestep in each trajectory,
		compute the return, and the
		advantage estimate (return  - the baseline b of the current state s_t)
	Optimize the baseline by minimizing ||baseline of current state - return||^2
	Update the policy using a policy gradient estimate g that is described below:
```

$\nabla_{\theta} log\pi(a_{t}|s_{t},\theta)\hat A_{t}$

### Intuitive Look at Objective Function

Objective function is intuitively satisfying. Let's see why:

Suppose we have a positive advantage estimate. Then the actions the agent took in the sample trajectory resulted in better than average return. This leads to a positive gradient and an increase in the probabilities of these actions.

Now suppose advantage estimate is negative. Then actions agent took in sample trajectory resutled in worse than average return, leading to negative gradient and decrease in probabilities of these actions.

### The Issue of Continually Running Gradient Descent on One batch of collected experience

Because there is no experience replay buffer like in Q-learning, continuously updating parameters via gradient descent on one batch of collected experience will result in updating parameters far outside the range of where the data was collected.

Take the advantage function, which is basically a noisy estimate of the real advantage. In the above scenario the advantage fucntion will be completely wrong.

Bottom Line:
Continuasly running gradient descent on a single batch of collected experience will destroy the policy. This is a problem. An idea for solving this is making sure that when you are updating policy, you don't move to far away from old policy. (See Trust Regions).

## Introduction of Trust Regions (TRPO)

Comparing the objective function of TRPO with Vanilla Policy Gradient:

Vanilla: $\hat E_{t}[log\pi_{\theta}(a_{t}|s_{t})\hat A_{t}]$

TRPO: $\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\hat A_{t}$

The only change here is that the log operator is replaced by dividing by $\pi_{\theta_{old}}$, thereby causing any new updates to be weighed down by old policy.

This isn't complete TRPO however. In fact, the equation above actually has the same gradient as the vanilla policy gradient objective function. Instead, TRPO sets up a **Lagrange Multiplier Optimization Problem** where we try to 

​	maximize  $\hat E_{t}[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\hat A_{t}]$
​	subject to $\hat E_{t}[KL[\pi_{\theta_{old}}(*|s_{t}),\pi_{\theta}(*|s_{t})]] \leq \delta$

Alternatively, one could use a penalty instead of a constraint ($expression \leq \delta$). They are equivalent mathematically. In practice using the constraint is easier to tune and a fixed $\delta$ is better than a fixed penalty $\beta$. 

In English, TRPO is summed up as "I want to optimize this objective (our local approximation) subject to some constraint that we are not moving too far away from the starting point." There are many ways to express this constraint, and we could simply use L2 (euclidean) distance to do it, but in general TRPO methods use the **KL divergence** between the old policy $\pi_{\theta_{old}}$ and the new policy $\pi_{\theta}$.

Using KL divergence instead of distance allows us to look at the distributions that the parameter vectors determine rather than just looking at the parameter vector. Just looking at parameter vectors doesn't give a good indication of what direction to update. KL divergence is a unit-less quantity and the constraint is that the action probability shouldn't change too much.

Overall Pseudocode for TRPO:

```
for iteration 1,2,... do
	Run policy for T timesteps or N trajectories
	Estimate advantage function at all timesteps
	Maximize TRPO objective function (above) constrained by
		KL divergence between current and old policy
end for
```

TRPO solves the constrained optimization problem efficiently by using the **conjugate gradient**.

F is Hessian

## Proximal Policy Optimization (PPO)