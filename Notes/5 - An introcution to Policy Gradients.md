# 5 - An introcution to Policy Gradients

In policy based methods, instead of learning a value function that tells us the expected sum of rewards for a given state-action pair, we directly learn the policy function: $\pi(s|a)$. Thus, we are selecting actions without using a value function.

We can use a value function to optimize the policy parameters, but the value function will not be used to select an action.



## Why Using Policy Based Methods?

### Two Types of Policy

Either **deterministic** or **stochastic**.

Deterministic policies map states to actions. Given a state it determines an action to take.

![img](https://cdn-images-1.medium.com/max/800/1*NDEGtK42rEpYLkTPg2LBPA.png)

Deterministic policies are used in deterministic environments. Taking an action means you are completely sure that it will happen. For example, moving a pawn in chess is an action that once taken, is sure to happen.

Stochastic Policies output a probability distribution over actions.

![img](https://cdn-images-1.medium.com/max/800/1*YCABimP7x1wZZZKqz2CoyQ.png)

Instead of being sure of taking some action $a$ (like moving left), there is a probability that we'll take a different one.

Stochastic policies are used when the environment is uncertain. This process is a **Partially Observable Markov Decision Process (POMDP)**.

Stochastic policies are more popular than deterministic ones.

### Advantages

Why should we use policy gradeints instead of deep q learning? There are three major advantages:

#### Convergence

Policy gradients have better convergence properties. As seen previosuly, value-based methods have a big oscillation while training. This comes from the fact that a choice of an action may change dramatically for an arbitrality small change in the estimated action values.

With policy gradient, however, optimization is simply done by following the gradient to find the best parameters. There is a smooth update of the policy at each step.

Because we follow the gradient to find the best parameters, we're guaranteed to converge on a local maximum (worst case) or global maximum (best case).

![img](https://cdn-images-1.medium.com/max/800/1*0lYcY5TBSqfNwdu8TduB6g.png)

#### Policy gradients are more effectiive in high dimensional action spaces

Policy gradients are more effective in high dimensional actions spaces (or when using continuous actions).

Issue with Deep Q-Learning is that predictions assign a score (that maximum expected future reward) for each possible action, at each time step, given the current state.

But what if there is an infinite possibility of actions? This is where Q-learning fails because we'd have to output a Q-value for each possible action.

With policy-based methods, you simply adjust the parameters directly, and understand what the maximum will be instead of estimating the maximum directly at every step.

![img](https://cdn-images-1.medium.com/max/800/1*_hAkM4RIxjKjKqAYFR_9CQ.png)

#### Policy Gradients can learn stochastic processes

Policy gradients can learn a stochastic policy, while value functions can't. This has two consequences.

First off, there is no longer a tradeoff between exploration and exploitation. Because a stochastic policy outputs a probability distribution over actions rather than the single action to take, the agent is able to explore the state space without always taking the same action. No need to hardcode exploration/exploitation with a hyperparameter like epsilon $ (\epsilon)$

Another problem that we get rid of is **perceptual aliasing**. This occurs when there are two states that seem to be (or actually happen to be) the same, but need different actions.

### Disadvantages

One big disadvantage of policy gradients is that they commonly converge to a local maximum rather than the global maximum. While Deep Q-Learning always tries to reach the maximum, policy gradients converge slower and take longer to train. However, there are solutions to this.

## Policy Search

We have our policy $\pi$ that has a parameter $\theta$. This outputs a probability distribution of actions:

$\pi _{\theta}(a|s)=P[a|s]$

To see if our policy is any good, we must find the best parameters for it ($\theta$). We will do this by maximizing a score function $J(\theta)$:

$J(\theta)=E_{\pi\theta}[\sum\gamma r]$

There are two steps:

1. Measure the qulity of a $\pi$ (policy) with a policy score function $J(theta)$
2. Use gradient ascent to find the best parameter $\theta$ that improves our $\pi$

###Step 1: Score the Policy $\pi$

In an episodic environment we use some start value. From there, we calculate the mean of the return from the first time step (G1). This is the cumulative discounted reward for the entire episode. This takes advantage of the fact that new episodes always start at the same state.

![img](https://cdn-images-1.medium.com/max/1400/1*tP4l4IrIG3aMLTrMt-1-HA.png)

In a continuous environment, we use the average value because we can't rely on a pecific start state.

Because some states happen more than others, each state is weighted.

![img](https://cdn-images-1.medium.com/max/1600/1*S-XLkrvPuVUqLrFW1hmIMg.png)

We also use the average reward per time step because we want to get the most reward per time step.

![img](https://cdn-images-1.medium.com/max/1600/1*3SejRRby6vAnThZ8c2UaQg.png)

### Step 2: Policy Gradient Ascent

Now we have a policy score function that tells us how good our policy is. Now we need to find parameters $\theta$ that maximize this score function. We use gradient ascent instead of descent because we want to find the maximum score, not the minimum error.

General idea:

![img](https://cdn-images-1.medium.com/max/800/0*oh-lF13hYWt2Bd6V.)

Implementing this mathematically is a little hard.

The best parameters $\theta^{*}$ that maximize the score can be shown as:
![img](https://cdn-images-1.medium.com/max/1200/1*xoGZI5v6lBS8s5OtBteJMA.png)

Score function defined as:

![img](https://cdn-images-1.medium.com/max/1000/1*dl4Fp0Izhv6bC0-qgThByA.png)

To do gradient ascent, we differentiate this function.  But this is actually quite hard. To see why, re-write score function as follows:

![img](https://cdn-images-1.medium.com/max/1600/1*qySDorYr55KgVJ6H3bu_6Q.png)

We know policy parameters affect how actions are chosen. So as a consequence, policy parameters affect what rewards we get, and ultimately what states we will see and how often. Because our performance depends on action selections and the distriution of states in which those seleciton are made AND both of those are affected by policy parameters finding the effect of the policy on the state distribution is unclear. In essence, the function of the environment is unknown.

The problem is estimating the $\nabla$ (gradient) with respect to the policy $\theta$ when the gradient depends on the unknown effect of policy changes on the state distribution.

Solution is known as the **Policy Gradient Theorem**.This provides an expression for $\nabla$ of $J(\theta)$ (performance/score) with respect to policy $\theta$ that does not involve the differentiation of the state distribution.

![img](https://cdn-images-1.medium.com/max/1000/1*dl4Fp0Izhv6bC0-qgThByA.png)

![img](https://cdn-images-1.medium.com/max/1600/1*i72jd_Hrimu9Aag70WGDmQ.png)

Because we are dealing with stochastic policy, we have to differentiate a probability distribution $\pi(\tau;\theta)$ that outputs the probability of taking a series of steps (s0, a0, r0...) given the current policy parameters $\theta$.

As seen elsewhere in machine learning, we never try to differentiate probability functions, instead we transform them into log likelihoods that are easily differentiable.

Ref: http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/

![img](https://cdn-images-1.medium.com/max/1600/1*iKhO5anOAfc3oqJOM2i_8A.png)

Then convert the summation back into an expectation function:

![img](https://cdn-images-1.medium.com/max/1000/1*4Y7BwUu2JBRIJ8bxXkzDjg.png)

And now we only need to compute the derivative of the log policy function. Once this is complete, we can do gradient ascent.

![img](https://cdn-images-1.medium.com/max/800/1*zjEh737KfmDUzNECjW4e4w.png)

The above policy gradient is telling us how we should shift the policy distribution through the changing of the parameters $\theta$ if we want to achieve a higher score.

$R(\tau)$ is a scalar value score:

- if it is high, then on average we took actions that lead to high rewards. We want to push the probabilities of the actions seen
  - So increase the probability of taking these actions
- However, if it is low we want to reduce probability of taking those actions.

This policy gradient causes the parameters to move most in the direction that favors highest rewards.

## Monte Carlo Policy Gradients

We use Monte Carlo in RL when tasks can be divided into episodes.

General algo:

```
Initialize 0
for episode in episodes tau = S0, A0, R1, S1, ..., ST:
	for t in range(1, T-1):
		calculate change in parameters by multiplying learning rate by policy gradient
		actually change the parameters
		
for each episode:
	At each time step within episode:
		Compute log probabilities produced by policy
		Multiply these log proabilities by the score function
		Update weights
```

Issue: we only calculate R at the end of the episode. So if we had a few bad actions by overall good actions, average will still be high and bad actions will get reinforced. 

Therefore, to have a correct policy, we will need a lot of samples, which results in slow learning.

### How to improve model?

Actor Critic: hybrid between value-based and policy-based algorithms

Proximal Policy Gradients: ensures deviation from previous policy stays small.
