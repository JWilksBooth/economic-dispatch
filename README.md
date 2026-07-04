# economic-dispatch

Single-turn power-systems economic dispatch RL environment for the `verifiers` library / Prime Intellect Environments Hub. (v0.1.1 — faithful rebuild of the June 2026 v0.1.0 design.)

## Task
Given online generators — each with a linear cost ($/MWh) and operating range [Pmin, Pmax] — and a system load (MW), output the least-cost dispatch as JSON: `{"G1": 12.3, "G2": 80.0, ...}` such that total generation equals load and every unit respects its range.

## Ground truth
For linear costs with box constraints and one balance equality, the optimum is the closed-form merit-order fill (commit every Pmin, then buy cheapest headroom first). Cross-checked against `scipy.optimize.linprog` (HiGHS): **200/200 instances, 0 mismatches.**

## Rewards
| Function | Measures | Weight |
|---|---|---|
| `reward_format` | Parseable JSON naming exactly the right units | 0.10 |
| `reward_power_balance` | sum(MW) matches load; linear falloff | 0.30 |
| `reward_limits` | Fraction of units within [Pmin, Pmax] | 0.20 |
| `reward_cost` | optimal/actual cost ratio, **hard-gated on feasibility** | 0.40 |

The feasibility gate on `reward_cost` is the anti-reward-hacking mechanism: dumping all load on the cheapest unit scores 0 on cost regardless of how cheap it looks. Verified in `tests/test_validation.py` (4 gate tests, all passing).

## Usage
```python
from economic_dispatch import load_environment
env = load_environment(num_examples=300)
```
```bash
N_INSTANCES=200 python tests/test_validation.py
vf-eval economic-dispatch -m <model> -n 50
```

## Roadmap
v2 (separate package, `dcopf-grid-verifiers`): DC optimal power flow with network topology and line limits — measured 45% of instances make the network-unconstrained merit-order dispatch infeasible, defeating exactly the reasoning this v1 environment rewards. v3: N-1 contingency-secure dispatch.
