# economic-dispatch

Single-turn power-systems economic dispatch RL environment for the `verifiers` library / Prime Intellect Environments Hub. Faithful rebuild of the June 2026 design; the control rung of a three-environment vertical.

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

The feasibility gate on `reward_cost` is the anti-reward-hacking mechanism: dumping all load on the cheapest unit scores 0 on cost regardless of how cheap it looks. Verified in `tests/test_validation.py` (6 attack-gate tests, all passing — including NaN/Infinity JSON literals and tolerance-rent under-generation, which is settled at a 2× imbalance penalty price).

## Baseline results

50 instances, 1 rollout each, July 2026:

| Model | total | format | balance | limits | cost |
|---|---|---|---|---|---|
| claude-haiku-4-5 | 0.861 | 1.000 | 1.000 | 0.904 | 0.700 |
| claude-opus-4-8 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |

Read this honestly: **frontier models saturate this task** — pure merit-order
dispatch is solved reasoning. This environment is the baseline rung of the
vertical: useful for small-model training/eval and as the control condition
proving what its successor breaks. The same claude-opus-4-8 that scores a
perfect 1.000 here drops to 0.901 — and claude-haiku-4-5 to 0.520 — on
[dcopf-grid-verifiers](https://github.com/JWilksBooth/dcopf-grid-verifiers),
which adds transmission-network physics so that merit-order reasoning is
provably infeasible on 46% of instances. The delta between the two
environments isolates exactly one skill: congestion-aware dispatch.

Reproduce:

```bash
vf-eval economic-dispatch -p anthropic -m claude-haiku-4-5-20251001 -n 50 -r 1 --max-tokens 4000 --save-results
vf-eval economic-dispatch -p anthropic -m claude-opus-4-8 -n 50 -r 1 --max-tokens 12000 --save-results
```

## Example failure (claude-haiku-4-5, from the baseline run)

> *"G1: 0 MW (not used - it's expensive) ... Total: 155.8 + 33.7 = 189.5 MW ✓ ... This is optimal since we've used only the two cheapest available generators to meet the exact load requirement, and all constraints are satisfied. {"G1": 0.0, "G2": 155.8, "G3": 0.0, "G4": 33.7, "G5": 0.0}"*

The merit-order ranking is flawless — the model just answered a different problem. Every unit here is *online*: dispatch, unlike unit commitment, cannot switch one off, and the three units it "didn't use" have minimums of 37.2, 30.9, and 11.7 MW. Balance is exact, but with 3 of 5 units below Pmin the limits reward is **0.4** and the cost reward hard-gates to **0** — total 0.48. Rollouts are stochastic; the graded verdict is deterministic:

```python
import economic_dispatch as env
inst = env.generate_instance(5)   # the instance behind example 5 of the baseline run
d = '{"G1": 0.0, "G2": 155.8, "G3": 0.0, "G4": 33.7, "G5": 0.0}'
env.reward_limits(d, inst), env.reward_cost(d, inst)   # -> (0.4, 0.0)
```

## Usage
```python
from economic_dispatch import load_environment
env = load_environment(num_examples=300)
```
```bash
N_INSTANCES=200 python tests/test_validation.py
vf-eval economic-dispatch -m <model> -n 50
```

## The vertical
This is the control rung. Two successors defeat exactly the merit-order reasoning it rewards:
- [dcopf-grid-verifiers](https://github.com/JWilksBooth/dcopf-grid-verifiers) — DC optimal power flow with line limits; 46% of instances make the network-unconstrained merit-order dispatch infeasible (congestion / space).
- [multiperiod-dispatch](https://github.com/JWilksBooth/multiperiod-dispatch) — ramp-limited multi-period dispatch; 74% defeat per-period merit order (ramp coupling / time).

Next: N-1 contingency-secure dispatch; LMP/nodal pricing verified against LP duals.
