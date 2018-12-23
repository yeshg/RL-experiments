---
typora-root-url: ./img
---

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

From Numerical Optimization (a field of math), we can solve a KL Penalized Problem:

1. Make linear approximation to objective, quadratic approximation to KL divergence term
   - KL divergence has two arguments (the two probability distributions to find distance between)
   - At the start point, distance is minimized if keeping one argument fixed. From there it seems to grow quadratically.
   - Goal is to find matrix that defines this quadratic distance function.
2. Solution to the KL Penalized problem is below:

$\theta - \theta_{old} = \frac{1}{\beta} F^{-1}g$ where $F$ is the Fisher information matrix, $g$ is the policy gradient. F is the Hessian of L:

$F = \frac{\partial^{2}}{\partial^{2}\theta}\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})|_{\theta=\theta_{old}}$

However, computing Hessian is computationally expensive, as you have to compute n gradients if you have n parameters. Instead, we can use the conjugate gradient algorithm to solve the linear equation $F^{-1}g$ without explicitly finding the Hessian $F$. For more on this, refer to textbook on numerical optimization. This is called hessian-free optimization.

Overall, this is development of ideas:

1. First we suggest optimizing the surrogate loss $L^{PG}$ or $L^{IS}$, but we found that instead using KL divergence to constrain the size of the update is better
2. This created a harder optiization problem to solve (constrained vs unconstrained).
   - We note that doing a quadratic and affine (linear) optimization to this problem  we get an update step that looks like $F^{-1} g$ (where F is Fisher info matrix and g is policy gradient)
3. This is the same as something that was proposed before called Natural Gradient
   - Solved for approximately by using conjugate gradient method.

Alternative Approach: KL constraint can sometimes lead to undesirable trianing behavior, it owuld be better to include this extra constraint directly into our optimization objective. This is PPO.

## Proximal Policy Optimization (PPO)

Let's build the objective funciton for PPO.

First, we define the probability ratio $r_{t}(\theta) = \frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}$. This will be somewhere from 0 to 1 if the action is less likely in the new policy than in the old.

Recall objective function for TRPO: $\hat E_{t}[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\hat A_{t}]$. This is the same as $r_{t}(\theta)$ multiplied by the advantage function.

So far, we have the same TRPO objective function just in a more readable form.

Following is central objective function for PPO:

$L^{CLIP}(\theta) = \hat E_{t}[min(r_{t}(\theta)\hat A_{t}, clip(r_{t}(\theta), 1 - \epsilon, 1 + \epsilon)\hat A_{t})]$

Let's break it down. Firstly, we clearly see that the objective function that PPO optimizes is an expectation operator (so we are going to compute the objective function over batches of trajectories), hence the $\hat E_{t}$.

The Expectation operator is taken over the minimum of two terms:

- $r_{t}(\theta)\hat A_{t}$
- $clip(r_{t}(\theta), 1 - \epsilon, 1 + \epsilon)\hat A_{t}$

First term is our probability ratio $r_{t}(\theta)$ multiplied by advantage function. This is the default objective for policy gradients which pushes the policy towards actions that yeild a high positive advantage over the baseline.

Second term is very similar, but it contains a truncated version of our probability ratio $r_{t}(\theta)$. This is done by the clip() function between $1-\epsilon$ and $1+\epsilon$ where $\epsilon$ is usually around 0.2 (it's a tunable hyperparameter).

The min operator is operated on both of these terms to get the final result. This min operator has a crucial relationship with the sign of our advantage estimate $\hat A_{t}$.

Picture below shows the two cases that A might take (positive or negative, if 0 then no update):

![](/ppo_min_explanation.png)

If the action was good, the advantage estimate is positive, AND if the same action already got updated in the previous step, we don't keep updating it too much or else it might get worse.

If the action was bad, the advantage estimte will be negative, AND if the same action just became more probable in the previous step, we want to undo the last update.

IN essence, PPO does same thing as TRPO - forces policy updates to be conservative if they move very far away from the current policy.

The simple PPO objective function often outperforms the more complicated one in TRPO.

### Rest of the PPO algorithm

There are two alternating threads in PPO.

In the first thread, the policy is interacting with the environment creating episode sequences for which we immediately calcualte the advantage estimates using our fitted baselines estimate for the state values.

Every so many episodes, a second thread is going to collect all that experience and run gradient descent on the policy network using the clipped PPO objective.

Below is OpenAI 5 training setup:

![](/openai5.png)

The two threds can be decoupled from each other by using a recent copy of the model (pink data-store) that is given to thousands of remote workers (top-left blue blocks), and a GPU cluster (green block) that runs gradient descent on the network.

The final loss function is the sum of the clipped PPO objective function described earlier plus two additional terms:

$L_{t}^{PPO}(\theta) = \hat E_{t}[L_{t}^{CLIP}(\theta) - c_{1}L_{t}^{VF}(\theta) + c_{2}S[\pi_{\theta}](s_{\theta})]$

The first term: $c_{1}L_{t}^{VF}(\theta)$ is in charge of updating the baseline network (so estimating how good or bad it is to be in this state). Mathematically, what is the aveage amount of discounted reward that I expect to get?

![](/ppo_actor_critic_style.png)

The value estimation portion of the network (left side) shares a large portion of its parameters with the policy network. As a result, the feature extraction portions of these networks are shared.

The second term: $c_{2}S[\pi_{\theta}](s_{\theta})$ is in charge of ensuring that we explore the environment (recall the exploration/exploitation tradeoff). This term is an entropy term that pushes the policy to behave more randomly until other parts of the objective start dominating.

Hyperparameters $c_{1}$ and $c_{2}$ determine how each term weigh into the loss.

In contrast to discrete aciton policies that output the action choice probabilities, the PPO head outputs the parameters of a Gaussian distribution for each available action type. When running the agent in training mode, it will sample from these distributions to get a continuous output value for each action head.

## Conclusion

PPO was developed to get rid of a lot of the very nasty code in a lot of other algorithms. It's much easier to tune than other alternatives.

It has the stability and reliability of TRPO while being much easier to implement.