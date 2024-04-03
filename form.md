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

---
### POMDP (partial observed Markov decision problems)

$M = (\chi, A, Z, \{P_a\}, \{O_a\}, c)$

- Its state space, $\chi$
- Its action space, $A$
- Its observation space, $Z$
- Its transition probabilities, \{P_a\}, $a \in A$
- Its observation probabilities, \{O_a\}, $a \in A$
- The immediate cost function, $c$

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


