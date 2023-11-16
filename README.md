# POLICY EVALUATION

# AIM

The aim of this experiment is to perform policy evaluation on two different policies within the Slippery Walk Five MDP environment. We will evaluate the effectiveness of these policies by calculating their state-value functions and comparing their performance.


# PROBLEM STATEMENT

In the Slippery Walk Five MDP environment, we have the task of navigating a slippery path from the starting state to a goal state while avoiding obstacles. The problem is to determine which policy, among the first and second policies, is more effective in achieving this goal. We will use policy evaluation to assess the performance of these policies.



# POLICY EVALUATION FUNCTION

The policy evaluation function takes as input a policy `pi`, the environment dynamics `P`, and optional parameters `gamma` (discount factor) and `theta` (convergence threshold). It computes the state-value function `V` for the given policy by iteratively updating the value estimates until they converge. The state-value function represents the expected cumulative rewards when following the policy `pi` from each state.

```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for a, transitions in P[s].items():
                for prob, next_s, reward, _ in transitions:
                    V[s] += pi(s) == a * (prob * (reward + gamma * prev_V[next_s]))
        if np.max(np.abs(V - prev_V)) < theta:
            break
        prev_V = V.copy()
    return V
```
# <br>OUTPUT

![exp213](https://github.com/obedotto/rl-policy-evaluation/assets/75243072/b3aa0de3-b756-4cf6-afd9-274dd162ba2e)


![exp21](https://github.com/obedotto/rl-policy-evaluation/assets/75243072/e7d2f05a-c4f4-4718-8b0e-97e55937151e)





# RESULT

After evaluating both policies using policy evaluation, it can be observed that both policies have the same state-value function, resulting in identical value estimates for all states in the Slippery Walk Five MDP environment. Therefore, there is no significant difference in performance between the first and second policies in terms of achieving the goal state and obtaining rewards. Both policies seem to have similar merits in this particular environment.

This experiment demonstrates that policy evaluation can be a useful tool for assessing the effectiveness of different policies in a Markov Decision Process.
