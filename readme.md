# Reinforcement Learning with Proximal Policy Optimization (PPO) on control of Custom Continuous Action Space Environment

## The Algorithm / Technique:

 PPO happens to be an on-policy technique, as it requires no target network to estimate the optimal policy. PPO is an improved version of the TRPO approach that aims to maximise the objective:

$L\_{theta} = \hat{E}\_t [\frac{\pi\_{\theta}(a\_t | s\_t)}{\pi\_{\theta old}(a\_t | s\_t)} \cdot \hat{A}\_t]$ eq(1)

Subject to: 

$\hat{E}\_t [KLD(\pi\_{\theta old}(a\_t | s\_t), \hspace{1mm} {\pi\_{\theta}(a\_t | s\_t)})] \leq \delta$ eq(2)

Where: 
$\hat{E}\_t$ corresponds to the expected value, $\pi\_{\theta old}(a\_t | s\_t)$ and $\pi\_{\theta}(a\_t | s\_t)$ correspond to the old and new policy estimates and $\hat{A}\_t$ corresponds to the GAE, expressed as:

$\hat{A}\_t = \sum_{i=t}^{T-1} \hspace{2mm} (\gamma \cdot \lambda)^{i-t} \cdot \hat{\delta}\_i$  eq(3)

$\hat{\delta}\_t = r\_t + \gamma \cdot \pi\_{\theta V}(s\_{t+1}) - \pi\_{\theta V}(s\_t)$  eq(4)

Where: 
$r\_t$ is the reward at time $t$, $\gamma$ is the discount factor, $\hat{\delta}\_t$ is the discounted return at time $t$, $\lambda$ is the decay factor for prior returns to the return at time $t$ and $\pi\_{\theta V}$ is the value model responsible for estimating the value return of a state at any given state in time.


Let: 
$r\_t(\theta) = \frac{\pi\_{\theta}(a\_t | s\_t)}{\pi\_{\theta old}(a\_t | s\_t)}$  eq(5)

Then:
$L\_{\theta} = \hat{E}\_t [r\_t(\theta)  \cdot \hat{A}\_t]$  eq(6)


It can be observed that eq(1) is a constrained optimization problem as it is subject to a constraint which is eq(2). Constrained Optimization problems are not particularly as easy to solve as unconstrained ones, typically they would require a conjugate gradient which at times can be more computationally expensive to compute and propagate. For this reason the PPO is a more preferred reinforcement learning approach.

In PPO, we aim to maximise the objective:

$L\_{\theta} = \hat{E}\_t [min(r\_t(\theta) \cdot \hat{A}\_t , \hspace{1mm} clip(r\_t(\theta), \hspace{1mm}1-\epsilon, \hspace{1mm} 1+\epsilon) \cdot \hat{A}\_t )]$  eq(7)

Note:
Maximising the objective $L\_{\theta}$ is also equivalent to minimising the objective $-L\_{\theta}$

The first term inside the $min$ is the CPI objective, using this objective alone, the policy model is almost certain to not converge as its current policy would tend to deviate largely from the old policy. To ensure convergence, the PPO paper utilises a clipped surrogate function to the CPI function, governed by an $\epsilon$ hyperparameter. By doing so, the model is certain to converge, while also simultaneously ensuring that the new policy is in close proximity to the old policy, thus the name “Proximal Policy Optimization”.

Having said all that, what exactly is $\pi\_{\theta}(a\_t | s\_t)$? well they do differ depending on whether the action space is discrete or continuous. For a discrete action space, $\pi(a\_t | s\_t)$ refers to the probability density function of a sampled action $a\_t$ in a categorical distribution defined by predicted logits for each corresponding action. 

Mathematically, given the model outputed logits: $(q\_1, q\_2, \ldots q\_n)$ , we can convert them to mutually exclusive probability scores $(p\_1, p\_2, \ldots p\_n)$ via a softmax function like so:
s
$(p\_1, p\_2, \ldots p\_n) = Softmax(q\_1, q\_2, \ldots q\_n)$

The mutually exclusive probability scores are then used to generate a categorical distribution, where we randomly sample discrete actions $a\_t$, where: 

$a\_t \sim Cat(p\_1, p\_2, \ldots, p)$

We then compute the log probability density of $a\_t$ given the distribution $Cat(p\_1, p\_2, \ldots, p)$ which we use as $\pi\_{\theta}(s\_t | a\_t)$. Since is in log probability form, the policy ratio $r(\theta)\_t$ is given as:

$r\_t{\theta} = \exp(\pi\_{\theta}(s\_t | a\_t) - \pi\_{{\theta} old}(s\_t | a\_t)) $ 

For a continuous action space, $\pi\_{\theta}(a\_t | s\_t)$ refers to the probability density function of a sampled from a normal or multivariate normal distribution defined by a predicted mean ($\mu$) and the corresponding predicted variance $\sigma^2$.

Mathematically, given the predicted mean and variance $\mu$ and $\sigma^2$, then we can randomly sample our continuous action from a normal distribution like so:

$a\_t \sim N(\mu, \sigma^2)$

Similar to the discrete action space, we compute the log probability density of $a\_t$ given the distribution $N(\mu, \sigma^2)$, this will then serve as our $\pi\_{\theta}$.

If we had a multi-dimensional action space, the distribution would become a multivariate normal distribution, where each sub distribution corresponds to a mean and variance of each action dimension in the action space
 
In our implementation, we used an $\epsilon$ value of $0.2$ and also added an “entropy bonus” as the original PPO paper suggested, the paper suggested that the entropy be maximised, as it increases the exploitativeness of the policy.

Ensure to browse through the code for the full implementation.


## The Environment

The environment is a single sinusoidal lane with upper and lower boundaries, each point in the siusoidal lane is governed by the equation:

$y = A \cdot \sin(\omega \cdot x)$
the left and right lanes are offset by the center lane by 1 and -1 respectively.

|   |   |
|----------|----------|
| Action Space     | Box(-0.5,0.5, (1,), float32)|
| Observation Space| Box(-3,3, (4,), float32)    |

The goal of the agent is to guide the vehicle to take only the sinusoudal path / lane.

### Action space

In this environment, the action corresponds to the derivative of the sinusoidal function that models the lane, and is expected to be estimated by the agent at each given state. The action state is multiplied by the common difference of the discrete x-axis values and added to the current state to give an approximate next state like so:

$s\_{t+1} = [\hat{y}\_{t+1, -1},\hspace{1mm} \hat{y}\_{t+1, 0}, \hspace{1mm} \hat{y}\_{t+1, 1}, \hspace{1mm} x\_{t+1}]$

Where:

$\hat{y}\_{t+1, 0} = \hat{y}\_{t, 0} + (A \cdot \omega \cos(\omega \cdot x) \cdot \Delta{x})$

$\hat{y}\_{t+1, -1} = \hat{y}\_{t+1, 0} - 1$

$\hat{y}\_{t+1, 1} = \hat{y}\_{t+1, 0} + 1$

And

$x\_{t+1} = x\_t + \Delta{x}$

The action in this case is: 
$a\_t = A \cdot \omega \cos(\omega \cdot x)$

So therefore:

$s\_{t+1} = s\_t + a\_t \cdot \Delta{x}$

In this environment, $-1 \leq a\_t \leq 1$


### Observation space

The observation space is a 3D space as described in the table:

|Num   |Obssevation   |Description   |
|--------|--------|--------|
|0       |$\hat{y}\_{t, -1}$| Position of left rear wheel in y-axis
|1       |$\hat{y}\_{t, 0}$| Position of center rear wheel in y-axis
|2       |$\hat{y}\_{t, 1}$| Position of right rear wheel in y-axis
|3       |$x\_t$| Position of center rear wheel in x-axis


### Reward

The reward function in this environment is simply: 

$r\_t = 1 - |\hat{y}\_{t, 0} - y\_{t}|$

Where:
$\hat{y}\_{t, 0}$ is the center rear wheel y-axis position estimated by the using policy action at time $t$ and $y\_t$ is the actual point in the y-axis is it ought to follow at time $t$ 


### Truncate and Terminate

Environment terminates if the end of the environment is reached of the agent exceeds the observation space bounds.

The environment truncates if the number of steps in the episode exceed 150
