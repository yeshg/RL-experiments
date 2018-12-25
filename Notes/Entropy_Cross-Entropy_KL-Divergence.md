# Introduction to Entropy, Cross-Entropy and KL-Divergence

These concepts come from Claude Shannon's Information Theory
(see his paper on the Mathematical Theory of Communication)

## Explaining Entropy

A bit is either 0 or 1. But not all bits are useful. Some are redundant.

In Shannon's theory, 1 useful bit of information reduced a recipient's uncertainty by a factor of 2.

For example, if weather is completely random, with 50% chance to be sunny or rainy. Now, if a weather station communicates to you that it will rain tommorow, your uncertainty has reduced by a factor of 2.

Regardless of how the weather channel comminicated this, only 1 bit of useful information was communicated.

If the weather has 8 possible options and the weather station tells you the weather tomorrow, the useful informaiton is in 3 bits:

$2^3 = 8$                          $log_{2}(8) = 3$ 

 But what if options are not equally likely (like 25% sunny, 75% rainy). Then the uncertinty reduction is just the inverse of the event's probability. So in this case:

$1/0.25 = 4$
$log_{2}(4) = 2$ so we have 2 bits of useful information

Now, $log(\frac{1}{x}) = -log(x)$ so the equation to compute the number of bits simplifie to:

$-log_{2}(probability)$

So, if the weather station tells us it will rain tommorow, we have $-log_{2}(.75) = 0.41$ bit of information.

How much do we get on average? 

$0.75 * 0.4$1 bits of information $ +$ $.25 * 2$ bits of information = $0.81$ bits of info on average. This is the **entropy.**

$H(p) = -\sum_{i}{p_{i}log_{2}(p_{i})}$

This is a measure of the the average amount of informaiton you get from one sample drawn from a given probability distribution $p$. It tells you how unpredictable $p$ is.

## Cross-Entropy

This is simply the average message length.

So going back to the example of eight possible options for weather, each being equally likely, the message length is 3 bits:

| 000  | 001         | 010           | 011    | 100        | 101           | 110        | 111           |
| ---- | ----------- | ------------- | ------ | ---------- | ------------- | ---------- | ------------- |
| sun  | partial sun | partial cloud | cloudy | light rain | moderate rain | heavy rain | thunder storm |

But what if we live in a sunny region where probability distribution is skewed towards sunny:

| 000  | 001         | 010           | 011    | 100        | 101           | 110        | 111           |
| ---- | ----------- | ------------- | ------ | ---------- | ------------- | ---------- | ------------- |
| sun  | partial sun | partial cloud | cloudy | light rain | moderate rain | heavy rain | thunder storm |
| 35%  | 35%         | 10%           | 10%    | 4%         | 4%            | 1%         | 1%            |

If we compute the entropy of this distribution, we find that it is 2.23 bits. Cross-entropy is still 3 bits.

We can do better.

| 00   | 01          | 100           | 101    | 1100       | 1101          | 11100      | 11101         |
| ---- | ----------- | ------------- | ------ | ---------- | ------------- | ---------- | ------------- |
| sun  | partial sun | partial cloud | cloudy | light rain | moderate rain | heavy rain | thunder storm |
| 35%  | 35%         | 10%           | 10%    | 4%         | 4%            | 1%         | 1%            |

We make sunny weather 2 bits, cloudy 3, light to moderate rain 4, and heavy rain+ 5 bits each. This coding is also unambiguous (chaining multiple messages together only leaves one way to interpret).

If we compute the new average number of bits sent, we get 

$35\%*2+35\%*2+10\%*3+...+1\%*5 = 2.42$ bits.

This isn't quite 2.23 bits (we can use other codes like the Huffman Code for more optimal lossless compression).

This code makes some implicit assumptions about the probability distribution. Instead of mostly sunny weather, if the probability dsitribution was heavily skewed towards rainy weather, the new code would actually be worse than the original, simply one (sending way more bits (cross-entropy) than what is useful (entropy)).

So, we can have a true distribution, $p$, and a predicted distribution $q$ which the code caters to.

Then, cross-entropy is:

$H(p,q) = -\sum_{i}(p_{i}log_{2}(q_{i}))$. Instead of using the true distribution like entropy, cross-entropy uses the predicted distribution $q$.

## Kullback-Leibler Divergence (KL Divergence)

If the distributions are the same, cross-entropy will equal entropy. If they are different, the cross-entropy will be greater than the entropy by some number of bits.

This distance is called the relative entropy, or the Kullback-Leibler Divergence (KL Divergence).

$D_{KL}(p||q) = H(p,q) - H(p)$

## How They are Used in ML

In a classification problem, which is supervised learning, we know the true distribution for all inputs.

If we are inputted a red panda, we know that the true distribution should be 100% for red panda, and 0% for everything else.

Our model, however, will predict its own distribution based on how the weights have been trained this far, and we could have a situation like below.

![](/Users/yeshg/RL/RL-experiments/Notes/img/cross_entropy_loss_example.png)

We can use the cross-entropy between $p$ and $q$ as a cost function here. This is called cross-entropy loss or log loss. In general, natural log is used instead of log base 2.

Cross-Entropy Loss:

$H(p,q) = -\sum_{i}(p_{i}ln(q_{i}))$

In this example, cross entropy is $-ln(0.25)$.