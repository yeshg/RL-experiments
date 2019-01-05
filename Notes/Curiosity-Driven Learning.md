# Curiosity-Driven Learning

Curiosity driven learning is the idea of building a reward function that is intrinsic to the agent. Essentially, the agent is a self-learner responsible providing its own feedback.

Curiosity driven learning agents perform as good as if they had extrinsic rewards, and were able to generalize bette with unexplored environments.

## Two Main Problems in Reinforcement Learning

### Sparse Rewards

Sparse rewards concern the time difference between an action and its feedback. An agent learns fast if each of its actions have a reward, and that agent gets rapid feedback. If we shoot and kill an enemy in space invaders, we are rewarded. The agent understands that this action at that state was good.

However, in more complex games the reward/feedback doesn't always come back immediately afterwards.

### Extrinsic Rewards Are Not Scalable

In these more complex environments, we have no way to scale a human-implemented reward function.

## Solution: A New Reward Function: *Curiosity*

Curiosity is an intrinsic reward that is equal to the error of our agent to predict the consequence of its own actions given its current state.

​	aka, the error of our agent to predict the next state given current state and action taken.

The idea of curiosity is encourage the agent to perform actions that reduce the uncertainty in the agent's ability to predict the consequence of its own action.

- uncertainty will be high in areas where the agent has spent less time or in areas with complex dynamics

To measure error, we need to build a model of environmental dynamics that predicts the next state given the current state and action

## The Intrinsic Curiosity Module

We are defining curiosity as the error between the predicted new state $s_{t+1}$ given our state $s_{t}$ and action $a_{t}$ and the actual new state.

But our state is a stack of 4 frames, so we are really asking to predict the next stack of frames, which is very hard:

- Hard to predict the pixels directly, if you're in Doom and move left you need to predict 248*248=61504 pixels
- It's probably not the right thing to do, predicting pixels is super hard, and there will always be a big pixel prediction error.
  - The agent will always be curious even if the feedback is unrelated to the agent's actions, therefore, continued curiosity is undersirable.

Instead of maing predictions in the raw sensor space (pixels), we need to transform the raw sensory input (array of pixels) into a feature space with only relevant information.

We define some rules for a good feature space:

- Needs to model things that can be controlled by the agent.
- Needs to also model things that can't be contolled by the agent but that can affect an agent.
- Needs to not model (and be unaffected by) things that are not in the agent's control and have no effect on him.

If our agent is a car, then we need to model:

- our car (controlled by the agent)
- other cars (not controlled, but still affect agent)

We don't need to model leaves on trees because they don't affect the agent and we don't control it.

This results in a feature representation with less noise. The desired embedding space should:

- be compact in terms of dimensionality (remove irrelevant parts of observation space like leaves)
- preserve sufficient info about observation.
- Stable: because non-stationary rewards make it difficult for reinforcement agents to learn.

Intrinsic Curiosity Module (ICM):

![image-20181225173643555](/Users/yeshg/RL/RL-experiments/Notes/img/icm.png)

The ICM generates curiosity reward, it is composed of two neural netowrks.

To learn the feature space from raw sensory input, we use **self-supervision**. This is training a network on a proxy inverse dynamics task of predicting the agent $\hat a_{t}$ given its current and enxt states ($s_{t}$ and $s_{t+1}$):

![image-20181225174725005](/Users/yeshg/RL/RL-experiments/Notes/img/icm_inverse_features.png)

Using the feature space we train a forward dynamics model that predicts the future representation of the next state $\hat \phi(s_{t+1})$ given the action we took $a_{t}$ and the feature representation of the current state $\phi(s_{t})$. We find the difference between $\hat\phi(s_{t+1})$ and $\phi(s_{t+1})$. This equals curiosity.

![image-20181225175558790](/Users/yeshg/RL/RL-experiments/Notes/img/icm_forward.png)

To summarize, there are two models in ICM:

- *Inverse Model* (blue): Encodes states $s_{t}$ and $s_{t+1}$ into feature vectors $\phi(s_{t})$ and $\phi(s_{t+1})$ that are trained to predict action $\hat a_{t}$.

$\hat a_{t} = g(s_{t},s_{t+1};\theta_{I})$ 
-> $g$ is the learning function (inverse model), $\theta_{I}$ are the parameters of $g$

Inverse Loss function: $min_{\theta_{I}}\,L_{I}(\hat a_{t},a_{t})$
It measures the difference between the real action and our predicted action 

- *Forward Model* (red): Takes encoded feature vectors and predicts feature representation

$\hat \phi(s_{t+1}) = f(\theta(s_{t},a_{t};\theta_{F})$
-> $f$ is the forward model with parameters $\theta_{F}$ and input $\phi(s_{t})$.

Forward Model Loss Function: $L_{F}(\phi(s_{t}),\hat \phi(s_{t+1})) = \frac{1}{2}|| \hat \phi(s_{t+1}) - \phi(s_{t+1})||^{2}$

Out of the ICM module comes the difference between predicted feature vector of the next state and the real feature vector of the next state:

$r_{t}^{i} = \frac{\eta}{2}||\hat \phi(s_{t+1} - \phi(s_{t+1}))||^2_{2}$

$r^{i}_{t}$ is the intrinsic reward at time $t$ (aka curiosity)
$\eta$ is the scaling factor that is greater than 0

The overall optimization problem on the ICM module is the composition of Inverse Loss and Forward Loss:

$min_{\theta_{P},\theta_{I},\theta_{F}}[-\lambda E_{\pi(s_{t};\theta_{P})}[\sum_{t}r_{t}]+(1-\beta)L_{I}+\beta L_{F}]$

## Recap of Curiosty-Driven Learning

- Because of sparse rewards and the inability to scale extrinstic rewards, we want to create an intrinsic reward for the agent
- We will do this through *curiosity*, which is the agent's error in predicting the consequence of its actions given its current state
  - This will push our agent to favor transitions with a high prediction error (higher in areas the agent has not explored much)
  - In this way, the agent explores the environment with curiosity
- But we can't predict the next state by predicting the next frame of pixels - that is too hard - instead we use feature representation that will only keep elements that
  - can be controlled by our agent
  - affect our agent
- To generate curiosity we use the *Intrinsic Curiosity Module* that is composed of two models:
  - Inverse Model - used to learn feature representation of the current and next state
  - Forward Dynamics Model - used to generate predicted feature representation of the next state
- Curiosity will be equal to the difference between the predicted feature vector of the next state and the actual feature vector of the next state

