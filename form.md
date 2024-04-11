## MC (Markov Chains)

#### Markov property

The state at instant t is enough to predict the state at instant t + 1

#### Assumptions

- There is only a finite number of possible states
- The probabilities do not depend on t

#### Stability

- **Irreducibility**

    A chain is irreducible if any state $y$ can be reached from any other
    state $x$.

- **Communicating classes**

    Every state in the subset is reachable from every other state in the subset.

    > *example:*
    >
    > <img src="communicating_classes.png" width="400" height="200">


- **Aperiodicity**

    $d_x = \gcd \{ t \in \mathbb{N} | P^t(x | x) > 0, t > 0 \}$
    $\gcd$ represents the greatest common divisor function

    A state $x$ is aperiodic if $dx = 1$
    > *example:*
    >
    > <img src="aperiodicity.png" width="400" height="200">


> **Key stability results**
>
> - An irreducible and aperiodic Markov chain possesses a
> stationary distribution.
>
> - For an irreducible and aperiodic Markov chain with stationary
>   distribution $\mu^*$,
>
>   $\lim_{t \to \infty} \mu_0 P^t = \mu^*$
>
>   for any initial distribution $\mu_0$.
>
> - If the chain is irreducible and aperiodic, it is ergodic.

## HMM (Hidden Markov models)

Model for sequential process with partial observability.

A HMM can be represented compactly as a tuple $(X, Z, P, O)$.

Matrix O is state by observations.

#### Estimation

- Filtering (Forward alghorithm)

    Given a sequence of observations, estimate the final state

- Marginal smoothing (Forward-backward algorithm)

    Given a sequence of observations, estimate some state in the middle

- Smoothing (Viterbi algorithm)

    Given a sequence of observations, estimate the whole sequence of states

- Prediction

    Given a sequence of observations, predict future states

    > We compute µT|0:T using the forward algorithm.
    >
    > Then we use the Markov property.\
    > $\boldsymbol{\mu}_{T+1 \mid 0: T}=\boldsymbol{\mu}_{T \mid 0:
    > T} \mathbf{P}$


#### Forward algorithm 

Given a sequence of observations $z_{0:t}$, the **forward mapping** $\alpha_t :
\mathcal{X} \to \mathbb{R}$ is defined for each $t$ as

$$
\alpha_t(x) = \mathbb{P}_{\mu_0}[X_t = x, Z_{0:t} = z_{0:t}]
$$

> How likely is it that I end up in $x$ having observed $z_{0:t}$

Given observation sequence $z_{0:T}$
1. Multiply initial distribution by $O(z_0|:)$

    equivalent to: $\alpha_0 =
    \text{diag}(O(z_0 | \cdot))\mu_0^T$
2. At each time step:

    a. Multiply current distribution by $P$

    b. Multiply by $O(z_0|:)$
    equilvalent to:
    $\alpha_t = \text{diag}(O(z_t | \cdot))P^T \alpha_{t-1}$

    > <img src="foward.png" width="400" height="200">
    >
    > don't forget to transpose $P$
3. Normalize \
    $\mu_{T|0:T} = \frac{\alpha_T}{\text{sum}(\alpha_T)}$

<img src="foward_alg_alpha_0.png" width="400" height="200">

#### Forward-backward algorithm

Given a sequence of observations $z_{0:t}$, the **backward mapping** $\beta_t :
\mathcal{X} \to \mathbb{R}$ is defined for each $t$ as

$$
\beta_t(x) = \mathbb{P}_{\mu_0} [Z_{t+1:T} = z_{t+1:T} | X_t = x]
$$

> How likely is it that I observe $z_{t+1:T}$ knowing that I'm in $x$

**Require**: Observation sequence $Z_{0:T}$

1. **Initialize**

    $\alpha_0 = \text{diag}(O(z_0 | \cdot))\mu_0^T$, $\beta_T = 1$  

2. **For** $\tau = 1, \ldots, T$ **do**

    $\alpha_{\tau} = \text{diag}(O(z_{\tau} | \cdot))P^T \alpha_{\tau-1}$  
    > Forward update

3. **end for**

4. **For** $\tau = T - 1, \ldots, 0$ **do** 

    $\beta_{\tau} = P\text{diag}(O(z_{\tau+1} | \cdot))\beta_{\tau+1}$  

    > *Backward update*
    >
    > (Note: first $\beta$ initialize all to 1)
5. **end for**

6. **Combine & normalize**

    $\alpha_t \odot \beta_t / \text{sum}(\alpha_t \odot \beta_t)$  

    > $\mu_{t \mid 0: T}(x)=\frac{\beta_t(x) \alpha_t(x)}{\sum_{y \in \mathcal{X}} \beta_t(y) \alpha_t(y)}$


#### Viterbi alghorithm

**Require**: Observation sequence $z_{0:T}$

1. **Initialize**

    $m_0 = \text{diag}(O(z_0 | \cdot))\mu_0^T$

2. **For** $\tau = 1, \ldots, T$ **do**
   
    $m_{\tau} = \text{diag}(O(z_{\tau} | \cdot)) \max [P^T \text{diag}(m_{\tau-1})]$
   
    $i_{\tau} = \arg \max [P^T \text{diag}(m_{\tau-1})]$

3. **end for**

4. **Let** $x^*_T = \arg \max_{x \in X} m_T(x)$

5. **For** $\tau = T - 1, \ldots, 0$ **do**
   
   $x^*_{\tau} = i_{\tau+1}(x^*_{\tau+1})$

6. **end for**

**return** $x^*_{0:T}$

> example
>
> <img src="vertebi_alg/1.png" width="300" height="200">
> <img src="vertebi_alg/2.png" width="300" height="200">
> <img src="vertebi_alg/3.png" width="300" height="200">
> <img src="vertebi_alg/4.png" width="300" height="200">
> <img src="vertebi_alg/5.png" width="300" height="200">
> <img src="vertebi_alg/6.png" width="300" height="200">
> <img src="vertebi_alg/7.png" width="300" height="200">
> <img src="vertebi_alg/8.png" width="300" height="200">
> <img src="vertebi_alg/9.png" width="300" height="200">
> <img src="vertebi_alg/10.png" width="300" height="200">
> <img src="vertebi_alg/11.png" width="300" height="200">
> <img src="vertebi_alg/12.png" width="300" height="200">
> <img src="vertebi_alg/13.png" width="300" height="100">

## Preferences 

$x$ is preferred to $y$, we write $x \succ y$

If an individual prefers outcome $x$ to $y$, that means that it
would be willing to pay some “fair amount” to have $x$ instead of
$y$


- It is anti-symmetric:
$x \succ y \implies y \not\succ x$
- It is negative transitive:
$x \not\succ y \text{ and } y \not\succ z \text{ then } x \not\succ z$
- If "≻" is a strict preference on some set of outcomes $(X)$

    - if "x ≻ y" or "x ~ y" we can write "x ≽ y" (x is not worse than y).

    - if "x ≺ y" or "x ~ y", we can write "x ≼ y" (x is not better than y).

Utility: u(x) ≥ u(y) if and only if x ≽ y.

## Taking decisions 

If she returns home…

- There is a 0.6 probability that she’ll arrive late at the
University, due to traffic

If she prints in the University…

- There is a 0.3 probability that she can’t find a printer in time
(she’ll submit an incomplete report)
- There is a 0.5 chance that the printer is busy (she’ll submit the
report late)

What are the possible outcomes?
- Report is complete and on time (CT, utility of 0)
- Report is complete but late (CL, utility of -2)
- Report is incomplete (IT, utility of -3)

<img src="taking_dec/decision_tree.png" width="400" height="300">
<img src="taking_dec/expected_values.png" width="400" height="300">
<img src="taking_dec/expected_values_2.png" width="400" height="300">

H ≻ U

## MDP (Markov decision problems)

$M = (\chi, A, \{P_a\}, c)$

- Its state space, $\chi$
- Its action space, $A$
- Its transition probabilities, \{P_a\}, $a \in A$
- The immediate cost function, $c$

#### Policies 

$h_t = \{x_0, a_0, x_1, a_1, ..., x_{t-1}, a_{t-1}\}$

- **Deterministic**

    - if there is one action that is selected with probability 1

        > **Does not choose actions randomly**
- **Markov**

    - if the distribution over actions given the history depends only on the last
    state (and t)

        > **Depends only on the last state**
- **Stationary**

    - if the distribution over actions given the history depends only on the last
    state (and not on t)

        > **Fixed through time**

#### $P_\pi$
<img src="p_pi.png" width="400" height="190">

#### $c_\pi$
<img src="c_pi.png" width="300" height="190">

#### $J^\pi$ (cost to go function per each initial state)

$$ J^\pi = c_\pi + \gamma P_\pi J^\pi $$
$$ J^\pi = (I - \gamma P_\pi)^{-1} c_\pi $$

##### Value iteration for $J^*$

$$ 
J^*(x) = \min_a \left[ c(x, a) + \gamma \sum_{y \in X} P_a(y | x)J^*(y) \right]
$$

<img src="value_iter/1.png" width="400" height="190">
<img src="value_iter/2.png" width="400" height="190">
<img src="value_iter/3.png" width="400" height="190">
<img src="value_iter/4.png" width="400" height="190">
<img src="value_iter/5.png" width="400" height="190">

#### $Q^\pi$ (cost-to-go for a fixed policy, given the initial state and action)

**Why should we care?**

We can compute cost-to-go functions from Q-functions:

$$ 
J^\pi(x) = \mathbb{E}_{a \sim \pi(x)} [Q^\pi(x, a)]
$$

> $J^\pi(x)$ is the element of each line of $Q^\pi(x, a)$ choosing the
> action by the policy .

$$ J^*(x) = \min_a Q^*(x, a) $$

> $J^*$ is the element of each line of $Q^*(x, a)$ choosing the
> action with the minium cost .

> Note:
> if $J^\pi(x) = \min_a Q^*(x, a)$, we can conclude that $J^\pi(x) = J^*(x)$
> and $\pi$ is optimal


We can compute the optimal policy directly from $Q^*$:

$$ \pi^*(x) = \underset{a}{\text{arg min }} Q^*(x, a) $$


<img src="q_pi.png" width="500" height="300">

Computation:
- Since
  $$ Q^\pi(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' \mid x)J^\pi(x') $$

- and
  $$ J^\pi(x) = \mathbb{E}_{a \sim \pi(x)} [Q^\pi(x, a)] $$

- then
$$ Q^\pi(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' \mid
x)\mathbb{E}_{a' \sim \pi(x')} [Q^\pi(x', a')] $$

##### Value iteration for $Q^*$
$$ Q^*(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' | x) \min_{a'}
Q^*(x', a') $$

#### Policy iteration
<img src="pol_iter/1.png" width="400" height="220">
<img src="pol_iter/2.png" width="400" height="220">

We stop this algorithm when the policy stabilizes, meaning that we found the
optimal policy.

#### Large problems
- **State aggregation**
    - States are “aggregated” into “chunks”
    - Each chunk is treated as a “super-state” 
    - Very limited representation power
- **Linear approximations**
    - States described by vector of “features” 
    - J and Q are represented as combinations of features
    - Good representation power; difficult to choose good features
- **Averagers**
    - Approximations that do not extrapolate
    - Such architectures are known as averagers

## POMDP (partial observed Markov decision problems)

$M = (\chi, A, Z, \{P_a\}, \{O_a\}, c)$

- Its state space, $\chi$
- Its action space, $A$
- Its observation space, $Z$
- Its transition probabilities, \{P_a\}, $a \in A$

    > Represents the probability of observing observation $j$ when action $a$ is
    > taken and the system is in the new state $i$.

- Its observation probabilities, \{O_a\}, $a \in A$
- The immediate cost function, $c$

#### Policies 

$h_t = \{z_0, a_0, z_1, a_1, ..., z_{t-1}, a_{t-1}\}$ (z is an observation not a
state)

The type of policies are equal to the **MDP** plus

- Memoryless:
    - if the distribution over actions given the history depends
only on the last observation

#### Forward alghorithm for POMDP

<img src="pomdp_old_trick/1.png" width="350" height="190">
<img src="pomdp_old_trick/2.png" width="350" height="190">
<img src="pomdp_old_trick/3.png" width="350" height="190">
<img src="pomdp_old_trick/4.png" width="350" height="190">
<img src="pomdp_old_trick/5.png" width="350" height="190">
<img src="pomdp_old_trick/6.png" width="350" height="190">

#### The belief

We call the distribution $\mu_{t|0:t}$ the belief at time t belief $b(t)$

When we want to calculate the $J^*$ we have a problem because we don't have
the actual state, we have an distribution $b(t)$, that's why we need to use
heuristics.

#### Heuristics for value iteration
- MLS

    Select the most likely state

- AV

    After choosing a an action for each state we sum all believes for the
    specific the bigger wins   

- Q-MDP

    $$ 
    J^*(b) = \min_{a \in A} \sum_{x \in \mathcal{X}} b(x) Q^*_{\text{MDP}}(x, a)
    $$
    > Weighted average of optimal MDP Q-values

- FIB heuristic

    $$
    Q_{\text{FIB}}(x, a) = c(x, a) + \gamma \sum_{z \in \mathcal{Z}}
    \min_{a' \in A} \sum_{x' \in \mathcal{X}} P_a(x' | x) O_a(z | x')
    Q_{\text{FIB}}(x', a')
    $$

    $$
    \pi_{\text{FIB}}(b) = \underset{a \in A}{\text{argmin}} \sum_{x \in
    \mathcal{X}} b(x) Q_{\text{FIB}}(x, a)
    $$
    > example
    >
    > <img src="heuristic_FIB.png" width="450" height="100">

#### Point based

Select a finite set sample of beliefs to perform updates

## Learning from examples in MDPs

We don't know the cost matrix neither the probability matrix

#### The inductive learning assumption
If we learn from a sufficiently large set of examples, we will do well in
the actual task.

- MDP-induced metric

    “clone” the observed behavior, using the MDP structure to generalize

- Inverse reinforcement learning

    “invert” the MDP 


#### Inverse reinforcement learning

**If someone shows the optimal policy, can we recover the task?** learn the cost function.

Agent is expected to recover the cost function implied by the policy $\pi^*$. 

An ill-posed problem refers to a problem that violates at least one of the
conditions:
- Existence: A solution exists.
- Uniqueness: The solution is unique.
- Stability: The solution's behavior changes continuously with the initial
  conditions or parameters. In other words, small changes in the input lead to
  small changes in the output. 

If we are given the optimal policy, then for all $x \in \mathcal{X}$ and all $a \in \mathcal{A}$,

$$J^*(x) \leq Q^*(x, a)$$

$$c_{\pi} + \gamma P_{\pi} J^* \leq c_a + \gamma P_a J^*$$

$$(P_{\pi} - P_a)(I - \gamma P_{\pi})^{-1}c \leq 0$$

Problem: all polices are optimal if c = 0 this is a ill-defined problem

#### Stochastic approximation (Interative learning)

- Model-based methods \
    the model trys to find the cost matrix and the probability matrix
- Value-based methods \
    the model trys to find the $Q^*$ and the $J^*$
- Policy-based methods \
    the model trys to find de policy

#### Model-based methods
We update the cost function and the transition probabilities with the mean
updated.

$$\hat{C}(x_t, a_t) = \hat{C}(x_t, a_t) + \alpha_t (c_t - \hat{C}(x_t, a_t))$$
$$\hat{P}(x' | x_t, a_t) = \hat{P}(x' | x_t, a_t) + \alpha_t (I[x_{t+1} = x'] -
\hat{P}(x' | x_t, a_t))$$

The model-based approach described converges to the true parameters P and c as
long as every state and action are visited infinitely often.

After each update we run VI update

$$J_{t+1}(x_t) = \hat{c}(x_t) + \gamma \sum_{x' \in \mathcal{X}} \hat{P}(x' |
x_t)J_t(x')$$

#### Monte Carlo RL

##### Computing $J^\pi$
- Given a trajectory obtained with policy $\pi$

$$
\tau_k=\left\{x_{k, 0}, c_{k, 0}, x_{k, 1}, c_{k, 1}, \ldots, c_{k, T-1},
x_{k, T}\right\}
$$
- Compute loss
$$
L\left(\tau_k\right)=\sum_{t=0}^{T-1} \gamma^t c_{k, t}
$$
- Update
$$
J_{k+1}\left(x_{k, 0}\right)=J_k\left(x_{k,
0}\right)+\alpha_k\left(L\left(\tau_k\right)-J_k\left(x_{k, 0}\right)\right)
$$

**Theorem**: For T large enough, Monte Carlo policy evaluation
converge w.p.1 to J , as long as every state is visited infinitely
often.

**Exploration in MC**: We can update the cost-to-go for every state visited along a trajectory
-  Consider only the first time a state appears: First-visit MC
-  Consider all the times a state appears: Every-visit MC


##### Computing $Q^\pi$
- Given a trajectory obtained with policy $\pi$
$$
\tau_k=\left\{x_{k, 0}, a_{k, 0}, c_{k, 0}, x_{k, 1}, a_{k, 1}, c_{k, 1},
\ldots, a_{k, T-1}, c_{k, T-1}, x_{k, T}\right\}
$$
- Compute loss
$$
L\left(\tau_k\right)=\sum_{t=0}^{T-1} \gamma^t c_{k, t}
$$
- Update
$$
Q_{k+1}\left(x_{k, 0}, a_{k, 0}\right)=Q_k\left(x_{k, 0}, a_{k, 0}\right)+\alpha_k\left(L\left(\tau_k\right)-Q_k\left(x_{k, 0}, a_{k, 0}\right)\right)
$$

#### $TD(\lambda)$

$$
\begin{gathered}
J_{t+1}(x)=J_t(x)+\alpha_t z_{t+1}(x)\left[c_t+\gamma J_t\left(x_{t+1}\right)-J_t\left(x_t\right)\right] \\
z_{t+1}(x)=\lambda \gamma z_t(x)+\mathbb{I}\left(x=x_t\right)
\end{gathered}
$$

- In this algorithm:
    - Each update uses informations from multiple steps
    - No looking in the future
    - No "long transitions" required

**Theorem:** For any $0 \leq \lambda \leq 1$, as long as every state is visited
infinitely often, TD($\lambda$) converges to $J^\pi$ w.p.1.


#### $Q$ learning

$$Q_{t+1}(x_t, a_t) = Q_t(x_t, a_t) + \alpha_t \left[ c_t + \gamma \min_{a'
\in A} Q_t(x_{t+1}, a') - Q_t(x_t, a_t) \right]$$

**Theorem:** As long as every state-action pair is visited infinitely often,
Q-learning converges to $Q*$ w.p.1.

#### Exploration vs exploitation

How can we visit every state action pair infinitely often?

- **Exploration**, i.e., trying new actions (or actions that have
been less experimented with)

- **Exploitation**, i.e., using the knowledge already acquired to
select the seemingly better actions

#### SARSA 

$$Q_{t+1}(x_t, a_t) = Q_t(x_t, a_t) + \alpha_t \left[ c_t + \gamma
Q_t(x_{t+1}, a_{t+1}) - Q_t(x_t, a_t) \right]$$

#### $Q$-learning vs SARSA

- Q-learning is an off-policy algorithm

    Learns the value of one policy while following another

- SARSA (like $TD(\lambda)$) is an on-policy algorithm

    Learns the value of the policy that it follows
    - For SARSA to learn Q* must be combined with policy improvement

#### On policy vs Off policy

- *On-policy* methods are typically simpler and more stable but might struggle with exploration. Leans the value of the policy that if follows.

- *Off-policy* methods, while more complex, can efficiently reuse experiences and potentially learn faster. Learns the value of one policy while following another.

#### Large domains

Monte Carlo methods **are** stochastic gradient descent methods Under mild
conditions, they do converge but May be stuch in **local minimium**


Now instead of having a matrix we have some parameters $\theta$ we would like to update 

> <img src="large_domais.png" width="250" height="100">
>
> $V$ is $J$ and $w$ is $\theta$

## Exploration vs Exploitation

- N = number of sources

- M = number of mistakes


### Weighted majority algorithm

- We measure our performance compared angaist that of the best "guess"

- Usually, performance of the best guess can only be assessed a posteriori

#### Algorithm

- Given a set of $N$ “predictors” and $\eta$ < 1/2

- Initialize predictor weights to $w_0(n) = 1, n = 1, ..., N $

- Make predictions based on the (weighted) majority vote

- Update weights of all wrong predictors as

$$w_{t+1}(n) = w_t(n)(1 - \eta)$$


### Exponentially Weighted Averager (EWA)

- Select each action “proportionally” to its confidence:

$$p_t(a)=\frac{w_t(a)}{\sum_{a^{\prime} \in \mathcal{A}} w_t\left(a^{\prime}\right)}$$

- When cost is revealed, we update each “confidence” according to the
  corresponding cost: 

$$w_{t+1}(a)=w_t(a) e^{-\eta c_t(a)}$$

- Then, at each step t,

$$w_t(a)=e^{-\eta \sum_{\tau=0}^{t-1} c_\tau(a)}$$

- It makes no assumptions on the process by which costs are selected (can be
  adversarial)

- Depends logarithmically on the number of actions (works well even if there is
  an exponentially large number of actions to try)

- Its regret is sublinear in T


### EXP3

- The diference is that in this scenario we do not observe the cost for all
actions, so we can only update weight for current action.

- Actions experimented more often will have more updates, which
will unbalance weights! So we compensate

#### Algorithm

- Given a set of $N$ actions and $\eta>0$

- Initialize weights to $w_0(a)=1, a \in \mathcal{A}$

- Select an action according to the probabilities

$$
p_t(a)=\frac{w_t(a)}{\sum_{a^{\prime} \in \mathcal{A}} w_t\left(a^{\prime}\right)}
$$

- Update weights of all actions as

$$
w_{t+1}\left(a_t\right)=w_t\left(a_t\right) e^{-\eta \frac{c_t\left(a_t\right)}{p_t\left(a_t\right)}}
$$


- It makes no assumptions on the process by which costs are selected (can be
  adversarial)  

- Depends sublinearly on the number of actions 

- Its regret is sublinear in T

### Stochastic bandits

- What is the goal? 
    - Select action with smallest average cost 

#### UCB algorithm

- Execute each action once

- From then on, at each step $t$, select action

$$
a^*=\underset{\substack{\text { Average } \\ \text { cost }}}{\operatorname{argmin}} \hat{c}(a)-\sqrt{\frac{2 \log t}{N_t(a)}}
$$

- Only observes cost for selected actions (bandit problem) 

- It assumes that costs follow an unknown (but fixed) distribution

- Its regret is sublinear in T

