# Formulário PADI

### MC
#### Key Property: Markov property
The state at instant t is enough to predict the state at instant t + 1

#### Assumptions
- There is only a finite number of possible states
- The probabilities do not depend on t

#### Stability
Irreducibility:\
A chain is irreducible if any state $y$ can be reached from any other state $x$

Communicating classes:\
<img src="communicating_classes.png" width="400" height="200">

Aperiodicity:\
$d_x = \gcd \{ t \in \mathbb{N} | P^t(x | x) > 0, t > 0 \}$

- $\gcd$ represents the greatest common divisor function\

A state $x$ is aperiodic if $dx = 1$

exmaple:\
<img src="aperiodicity.png" width="400" height="200">

Key stability results:
- An irreducible and aperiodic Markov chain possesses a
stationary distribution.
- For an irreducible and aperiodic Markov chain with stationary
distribution $\mu^*$,\
$\lim_{t \to \infty} \mu_0 P^t = \mu^*$\
for any initial distribution $\mu_0$.

---
### Hidden Markov models
Model for sequential process with partial observability

A HMM can be represented compactly as a tuple $(X, Z, P, O)$

#### Estimation
- Filtering (Forward alghorithm) \
Given a sequence of observations, estimate the final state

- Marginal smoothing (Forward-backward algorithm)\
Given a sequence of observations, estimate some state in the middle

- Smoothing (Viterbi algorithm) \
Given a sequence of observations, estimate the whole sequence of states

- Prediction\
Given a sequence of observations, predict future states\
We compute µT|0:T using the forward algorithm\
We use the Markov property


#### Forward alghorithm
Given observation sequence $z_{0:T}$
1. Multiply initial distribution by $O(z_0|:)$\
    equivalent to: $\alpha_0 =
    \text{diag}(O(z_0 | \cdot))\mu_0^T$
2. At each time step:\
    a. Multiply current distribution by $P$ \
    b. Multiply by $O(z_0|:)$ \
    equilvalent to :
    $\alpha_t = \text{diag}(O(z_t | \cdot))P^T \alpha_{t-1}$
3. Normalize \
    $\mu_{T|0:T} = \frac{\alpha_T}{\text{sum}(\alpha_T)}$

<img src="foward_alg_alpha_0.png" width="400" height="200">

#### Forward-backward algorithm
**Require**: Observation sequence $Z_{0:T}$

1. **Initialize**\
$\alpha_0 = \text{diag}(O(z_0 | \cdot))\mu_0^T$, $\beta_T =
   1$  
2. **For** $\tau = 1, \ldots, T$ **do** \
   $\alpha_{\tau} = \text{diag}(O(z_{\tau} | \cdot))P^T \alpha_{\tau-1}$  
   *Forward update*
3. **end for**
4. **For** $\tau = T - 1, \ldots, 0$ **do** \
   $\beta_{\tau} = P\text{diag}(O(z_{\tau+1} | \cdot))\beta_{\tau+1}$  
   *Backward update* \
   (Note: first $\beta$ initialize all to 1)
5. **end for**

**return** \
$\alpha_t \odot \beta_t / \text{sum}(\alpha_t \odot \beta_t)$  
*Combine & normalize*

#### Viterbi alghorithm

**Require**: Observation sequence $z_{0:T}$

1. **Initialize** \
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

<img src="vertebi_alg/1.png" width="400" height="200">
<img src="vertebi_alg/2.png" width="400" height="200">
<img src="vertebi_alg/3.png" width="400" height="200">
<img src="vertebi_alg/4.png" width="400" height="200">
<img src="vertebi_alg/5.png" width="400" height="200">
<img src="vertebi_alg/6.png" width="400" height="200">
<img src="vertebi_alg/7.png" width="400" height="200">
<img src="vertebi_alg/8.png" width="400" height="200">
<img src="vertebi_alg/9.png" width="400" height="200">
<img src="vertebi_alg/10.png" width="400" height="200">
<img src="vertebi_alg/11.png" width="400" height="200">
<img src="vertebi_alg/12.png" width="400" height="200">
<img src="vertebi_alg/13.png" width="400" height="200">

---
### Preferences 

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

Utility:\
u(x) ≥ u(y) if and only if x ≽ y.

---
### Taking decisions 

example:

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

---
### MDP (Markov decision problems)

$M = (\chi, A, \{P_a\}, c)$

- Its state space, $\chi$
- Its action space, $A$
- Its transition probabilities, \{P_a\}, $a \in A$
- The immediate cost function, $c$

#### Policies 

$h_t = \{x_0, a_0, x_1, a_1, ..., x_{t-1}, a_{t-1}\}$

- Deterministic:
    - if there is one action that is selected with probability 1\
**Does not choose actions randomly**
- Markov:
    - if the distribution over actions given the history depends only on the last
    state (and t)\
**Depends only on the last state**
- Stationary:
    - if the distribution over actions given the history depends only on the last
    state (and not on t)\
    **Fixed through time**

#### $P_\pi$
<img src="p_pi.png" width="400" height="190">

#### $c_\pi$
<img src="c_pi.png" width="300" height="190">

#### $J^\pi$ (cost to go function per each initial state)

$$ J^\pi = c_\pi + \gamma P_\pi J^\pi $$
$$ J^\pi = (I - \gamma P_\pi)^{-1} c_\pi $$

#### Value iteration for $J^*$

$$ J^*(x) = \min_a \left[ c(x, a) + \gamma \sum_{y \in X} P_a(y | x)J^*(y)
\right] $$

<img src="value_iter/1.png" width="400" height="190">
<img src="value_iter/2.png" width="400" height="190">
<img src="value_iter/3.png" width="400" height="190">
<img src="value_iter/4.png" width="400" height="190">
<img src="value_iter/5.png" width="400" height="190">

#### $Q^\pi$ (cost-to-go for a fixed policy, given the initial state and action)

**Why should we care?**\
We can compute cost-to-go functions from Q-functions:

$$ J^\pi(x) = \mathbb{E}_{a \sim \pi(x)} [Q^\pi(x, a)] $$
$$ J^*(x) = \min_a Q^*(x, a) $$

We can compute the optimal policy directly from $Q^*$:

$$ \pi^*(x) = \underset{a}{\text{arg min }} Q^*(x, a) $$

Computation:
- Since
  $$ Q^\pi(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' \mid x)J^\pi(x') $$
- and
  $$ J^\pi(x) = \mathbb{E}_{a \sim \pi(x)} [Q^\pi(x, a)] $$
- then
$$ Q^\pi(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' \mid
x)\mathbb{E}_{a' \sim \pi(x')} [Q^\pi(x', a')] $$

#### Value iteration for $Q^*$
$$ Q^*(x, a) = c(x, a) + \gamma \sum_{x' \in X} P_a(x' | x) \min_{a'}
Q^*(x', a') $$



#### Policy iteration
<img src="pol_iter/1.png" width="400" height="220">
<img src="pol_iter/2.png" width="400" height="220">

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

---
### POMDP (partial observed Markov decision problems)

$M = (\chi, A, Z, \{P_a\}, \{O_a\}, c)$

- Its state space, $\chi$
- Its action space, $A$
- Its observation space, $Z$
- Its transition probabilities, \{P_a\}, $a \in A$
- Its observation probabilities, \{O_a\}, $a \in A$
- The immediate cost function, $c$

#### Policies 

$h_t = \{z_0, a_0, z_1, a_1, ..., z_{t-1}, a_{t-1}\}$ (z is an observation not a
state)

the type of policies are equal to the **MDP** plus

- Memoryless:
    - if the distribution over actions given the history depends
only on the last observation

#### Forward alghorithm for POMDP

<img src="pomdp_old_trick/1.png" width="350" height="190">
<img src="pomdp_old_trick/2.png" width="350" height="190">
<img src="pomdp_old_trick/3.png" width="350" height="190">
<img src="pomdp_old_trick/4.png" width="350" height="190">
<img src="pomdp_old_trick/5.png" width="350" height="190">

#### The belief

We call the distribution $\mu_{t|0:t}$ the belief at time t belief $b(t)$

When we want to calculate the $J^*$ we have a problem because we don't have
the actualy state we have an distribution $b(t)$ that why we need to use
heuristics

#### Heuristics for value iteration
- MLS\
    Select the most likely state
- AV\
    After choosing a an action for each state we sum all believes for the
    specific the bigger wins   
- Q-MDP
    $$ J^*(b) = \min_{a \in A} \sum_{x \in \mathcal{X}} b(x) Q^*_{\text{MDP}}(x, a) $$
    Weighted average of optimal MDP Q-values
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

#### Point based
Select a finite set Bsample of beliefs to perform updates

---
### Inverse reinforcement learning

Agent is expected to recover the cost function implied by the policy $\pi^*$. 

The IRL problem is intrinsically ill-posed, as there are infinitely many
possible solutions for
each instance of the problem.

IRL problems:
- Multiple Reward Functions for the Same Behavior: There can be infinitely
  many reward functions that would make the observed expert behavior appear
  optimal. 
- Ambiguity in Observed Behavior: The observed behavior of an expert is
  typically a limited subset of all possible behaviors in the environment.

An ill-posed problem refers to a problem that violates at least one of the
conditions:
- Existence: A solution exists.
- Uniqueness: The solution is unique.
- Stability: The solution's behavior changes continuously with the initial
  conditions or parameters. In other words, small changes in the input lead to
  small changes in the output. 


