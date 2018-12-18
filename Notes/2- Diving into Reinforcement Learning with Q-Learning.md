**2- Diving into Reinforcement Learning with Q-Learning**



**Big Picture**

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/C0B4DE97-2A97-45E8-8B23-83A2569D3B13/Image.tiff)

Player is a knight (top left) that must avoid landing on a square with a red guard. Player wins by reaching the Princess in the castle (bottom middle). Player moves one tile at a time, and wants to reach castle via the fastest route possible. This all can be quantified with a point-system:

- Lose 1 point at each step (that way, agent is encouraged to reach princess asap)
- Touching enemy loses 100 points, and the game (episode) ends
- If castle is reached, you win with +100 points



**Naive Approach**

What if we simply make our agent try to go to each tile and then color each tile red (causes loss) or green (safe). Then, just tell agent to only take green tiles. Issue with this is that there is no sense of which tile is the best to take when green tiles are next to each other. In fact, agent could get stuck in an infinite loop.



**Introducing the Q-table**

A smarter strategy is to create a table where we will calculate maximum expected future reward, for each action at each state.  This way, we will have a sense for what action is the best to take for each state.



Each state (tile on the board) allows four possible actions (moving left, right, up, or down). At first this can be visualized as a grid, where each tile is divided into four possible actions. 0 means that a particular action is impossible (like moving out of bounds).

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/8AD3C1AD-6AAF-4679-8E3C-96642E84F331/Image.tiff)



This grid can then be transformed into a table:

![Screen Shot 2018-12-07 at 12.09.56 AM.png](/Users/yeshg/Library/Application Support/typora-user-images/C0B4DE97-2A97-45E8-8B23-83A2569D3B13/Screen%20Shot%202018-12-07%20at%2012.09.56%20AM.png)

This is called a Q-table (“Q” for quality of the action). The columns will be the four actions, and the rows will be the states. The value of the action/state cell will be the maximum expected future reward for that given state and action.



Each Q-table score will be the maximum expected future reward if that action is taken at that state with the **best policy given.**



Here, we say “best policy given” because we don’t actually implement one. Instead, we improve the Q-table to always choose the best action.



In this sense, Q-table is basically a cheat sheet.



Now, we have to learn each value of the Q-table using the **Q Learning Algorithm.**



**Q-Learning Algorithm: learning the Action Value Function**

The action value function (or “Q-function”) is a multivariable function on the state and action. It returns the expected future reward of that action at that state:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/3C28AE7C-1259-4DE4-ACFD-F1DF55F46175/Image.tiff)



Essentially, the Q-function looks through the !-table too find the line associated with the state, and the column associated with out action. Then it returns the Q value from the matching cell (which is the expected future reward).



Before exploring the environment, the Q-table gives the same arbitrary fixed value (usually 0). As we explore, Q-table iteratively updates and will give a better approximation. The Q-table is improved with the **Bellman Equation**.



![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/D4DCF7C6-DEEE-42F2-88FC-0CF349FF2B4D/Image.tiff)



1. Initialize Q-values (Q(s,a)) arbitrarily for all state-action pairs (usually initialize to 0)

2. For life or until learning stops:

3. 1. Choose action (a) in the current world state (s) based on current Q-value estimates (Q(s,*)).
   2. Take the action (a) and observe the outcome state (s’) and reward (r)
   3. Update following Bellman Equation

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/B03FBA33-6A2F-4440-9B93-4F3C09622BAF/Image.tiff)

Above is the basic pseudo-code for the Q-learning algorithm.



Step 1: Initialize Q-values

Build a Q-table of m x n columns and rows. m=number of actions, n=number of states. Initialize all values to 0:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/2F9ABD5F-8F4E-4851-8F5A-84E4AE443310/Image.tiff)

Step 2: Repeat steps 3-5 for life or until learning is stopped:

Exit step 2 once maximum number of episodes is reached (that max is specified by user) or until user manually stops training.



Step 3: Choose an Action

Choose an action *a* in the current state *s* based on the current Q-value estimates.



But if we initialize entire table to 0 in the beginning, what action do we take?



This is where exploration/exploitation comes in to play. We need to explore so our Q-table is useful.



Idea: In the beginning of training, use the **epsilon greedy strategy**:

- Define an exploration rate “epsilon” which is set to 1 at the beginning. This is essentially the rate of steps we’ll do randomly.

- - In the beginning, this value must be at its highest value, because we want to explore as much as possible to learn and make Q-table useful.

- Generate a random number. If this number > epsilon, do “exploitation” (using what we already know to select best action at each step). Else, we’ll do exploration.

- Idea is that we must have a big epsilon at the beginning of the training of the Q-function. Then, reduce it progressively over time as agent’s Q-table becomes better (agent is more confident at estimating Q-values).

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/9433D100-204D-4CB2-A680-329AA3197167/Image.tiff)

Steps 4-5: Evaluate

Take action *a* and observe the outcome states *s’* and reward *r*. Now update function Q(s,a).



Bellman Equation used to update Q(s,a) uses current Q value, learning rate, discount rate, reward for taking action *a* at state *s* and the maximum expected future reward given the new state *s’* and all possible actions at that new state:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/BDBF974A-7170-47E2-BE15-8274C987D5D3/Image.tiff)

So, our new Q value = Our current Q value + lr * [Reward + discount rate * (highest Q value between possible actions from the new state *s’*) - Current Q value]



Walking through an example:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/C83C9863-1E01-4E1B-AB58-9055D1564E8C/Image.tiff)

We are the mouse. The one cheese tile is +1 points. The two cheese tile is +2 points. The big pile of cheese is +10 points and the end of the episode. Eating rat poison is -10 points, the end of the episode.

Step 1: Init Q-table:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/01313CCD-7EAA-4D1D-8165-A087B5C0FC17/Image.tiff)

Step 2: Choose an Action

In beginning, we know nothing, but have big epsilon rate, so we take random move. Let’s say we move right:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/76AA27CE-2C66-4568-BB29-2ACA8295F3BE/Image.tiff)

Now we can update Q-value of being at the start and going right, using Bellman Equation.



Step 4-5: Update Q-function:

![Image.tiff](/Users/yeshg/Library/Application Support/typora-user-images/C3603481-1CA0-4A63-BA5C-B714DEE19A65/Image.tiff)



- First calculate change in Q value *deltaQ(start, right)*
- Then add initial Q value to the change in Q value multiplied by a learning rate.



Here, learning rate is how quickly the network abandons the earlier value for the new one. If our learning rate is 1, then new estimate will be the new Q-value.



Now do this repeatedly until learning is stopped.