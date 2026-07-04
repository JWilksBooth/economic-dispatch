"""economic-dispatch: single-turn merit-order dispatch RL environment (verifiers spec).

Rebuilt to the original v1 spec (June 2026):
- Units with linear cost [$/MWh] and range [Pmin, Pmax]; system load in MW.
- Ground truth: closed-form merit-order fill (commit every Pmin, then buy
  cheapest available headroom first). Exact for linear costs + box constraints
  + single balance equality.
- Cross-check: scipy.optimize.linprog on the same instances.
- Answer format: JSON dict mapping unit names to MW, e.g. {"G1": 12.3, "G2": 80.0}
- Rewards: format 0.10, power_balance 0.30, limits 0.20, cost 0.40
  (cost gated on feasibility — the anti-reward-hacking mechanism).
"""

from __future__ import annotations

import json
import random
import re

__version__ = "0.1.3"  # 0.1.0 was the lost June 2 build; logic identical to spec

DEFAULT_NUM_EXAMPLES = 300


# ---------------- instance generation ----------------

def generate_instance(seed: int) -> dict:
    rng = random.Random(seed)
    n = rng.randint(3, 6)
    units = []
    for i in range(n):
        p_max = round(rng.uniform(30, 200), 1)
        p_min = round(rng.uniform(0.0, 0.3) * p_max, 1)
        cost = round(rng.uniform(10, 90), 2)
        units.append({"name": f"G{i+1}", "cost": cost,
                      "p_min": p_min, "p_max": p_max})
    lo = sum(u["p_min"] for u in units)
    hi = sum(u["p_max"] for u in units)
    load = round(rng.uniform(lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)), 1)
    return {"units": units, "load_mw": load}


def instance_to_prompt(inst: dict) -> str:
    units_txt = "\n".join(
        f"  {u['name']}: cost ${u['cost']}/MWh, output range [{u['p_min']}, {u['p_max']}] MW"
        for u in inst["units"])
    example = ", ".join(f'"{u["name"]}": <MW>' for u in inst["units"])
    return f"""You are a power system operator performing economic dispatch.

Online generators:
{units_txt}

System load: {inst['load_mw']} MW

Find the least-cost dispatch: a MW output for every unit such that total
generation equals the load and every unit stays within its output range.

Report each MW value to at least one decimal place. Constraints are verified
with a +/-0.5 MW tolerance, but imbalance or bound violations inside that
tolerance are penalized in the cost scoring, so target exact balance.

Respond with your final answer as JSON on the last line, exactly:
{{{example}}}"""


# ---------------- ground-truth solvers ----------------

def merit_order_solve(inst: dict) -> dict[str, float]:
    """Closed-form: commit all Pmin, then fill cheapest headroom first."""
    dispatch = {u["name"]: u["p_min"] for u in inst["units"]}
    remaining = inst["load_mw"] - sum(dispatch.values())
    for u in sorted(inst["units"], key=lambda u: u["cost"]):
        if remaining <= 1e-9:
            break
        take = min(remaining, u["p_max"] - u["p_min"])
        dispatch[u["name"]] += take
        remaining -= take
    return dispatch


def linprog_solve(inst: dict) -> dict[str, float] | None:
    """Independent cross-check via scipy."""
    from scipy.optimize import linprog
    units = inst["units"]
    c = [u["cost"] for u in units]
    A_eq = [[1.0] * len(units)]
    b_eq = [inst["load_mw"]]
    bounds = [(u["p_min"], u["p_max"]) for u in units]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None
    return {u["name"]: float(x) for u, x in zip(units, res.x)}


def dispatch_cost(inst: dict, dispatch: dict[str, float]) -> float:
    cost_by_name = {u["name"]: u["cost"] for u in inst["units"]}
    return sum(cost_by_name[k] * v for k, v in dispatch.items())


# ---------------- parsing and rewards ----------------

def _is_finite_number(v) -> bool:
    """True for finite int/float; False for bool, NaN, Inf (json.loads accepts
    those literals; NaN defeats comparison-based checks), and integers too
    large for float (whose math.isfinite would raise OverflowError)."""
    import math
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def parse_answer(completion: str, inst: dict) -> dict[str, float] | None:
    names = {u["name"] for u in inst["units"]}
    for raw in reversed(re.findall(r"\{[^{}]+\}", completion, re.DOTALL)):
        try:
            obj = json.loads(raw)
        # ValueError covers JSONDecodeError AND CPython's >4300-digit int limit;
        # RecursionError covers deeply nested brackets. Both must skip, not crash.
        except (ValueError, RecursionError):
            continue
        if (isinstance(obj, dict) and set(obj.keys()) == names
                and all(_is_finite_number(v) for v in obj.values())):
            return {k: float(v) for k, v in obj.items()}
    return None


def _feasible(inst: dict, d: dict[str, float], tol: float = 0.5) -> bool:
    if abs(sum(d.values()) - inst["load_mw"]) > tol:
        return False
    return all(u["p_min"] - tol <= d[u["name"]] <= u["p_max"] + tol
               for u in inst["units"])


def reward_format(completion: str, inst: dict) -> float:
    return 1.0 if parse_answer(completion, inst) is not None else 0.0


def reward_power_balance(completion: str, inst: dict) -> float:
    d = parse_answer(completion, inst)
    if d is None:
        return 0.0
    err = abs(sum(d.values()) - inst["load_mw"])
    return max(0.0, 1.0 - err / max(inst["load_mw"], 1e-9) * 10)  # linear falloff


def reward_limits(completion: str, inst: dict) -> float:
    d = parse_answer(completion, inst)
    if d is None:
        return 0.0
    ok = sum(1 for u in inst["units"]
             if u["p_min"] - 0.5 <= d[u["name"]] <= u["p_max"] + 0.5)
    return ok / len(inst["units"])


def reward_cost(completion: str, inst: dict,
                optimal_cost: float | None = None) -> float:
    """optimal/actual ratio, HARD-GATED on feasibility.

    Tolerance-rent guard: _feasible allows +/-0.5 MW slack, so under-serving
    load (or riding past a bound) can make `actual` cheaper than the optimum;
    a bare min(1, opt/actual) clamp would score that a perfect 1.0. Residual
    violations are therefore priced at the most expensive unit's rate, and the
    ratio is made symmetric — a cost below the optimum (physically impossible
    for an honest feasible dispatch) scores below 1.0 too.
    """
    d = parse_answer(completion, inst)
    if d is None or not _feasible(inst, d):
        return 0.0
    if optimal_cost is None:
        optimal_cost = dispatch_cost(inst, merit_order_solve(inst))
    # Violations settled at a penalty price ABOVE the highest offer (2x), like
    # real imbalance settlement — at 1x, shaving the priciest unit is merely
    # cost-neutral instead of strictly losing.
    penalty_rate = 2.0 * max(u["cost"] for u in inst["units"])
    shortfall = max(0.0, inst["load_mw"] - sum(d.values()))
    bound_violation = sum(max(0.0, u["p_min"] - d[u["name"]]) +
                          max(0.0, d[u["name"]] - u["p_max"])
                          for u in inst["units"])
    actual = dispatch_cost(inst, d) + (shortfall + bound_violation) * penalty_rate
    actual = max(actual, 1e-9)
    return min(optimal_cost / actual, actual / optimal_cost)


REWARD_WEIGHTS = {"reward_format": 0.10, "reward_power_balance": 0.30,
                  "reward_limits": 0.20, "reward_cost": 0.40}


# ---------------- verifiers entry point ----------------

def build_dataset(num_examples: int = DEFAULT_NUM_EXAMPLES, seed_offset: int = 0):
    rows = []
    for i in range(num_examples):
        inst = generate_instance(seed_offset + i)
        opt = merit_order_solve(inst)
        rows.append({
            "question": instance_to_prompt(inst),
            "answer": str(round(dispatch_cost(inst, opt), 2)),
            "info": {"instance": inst,
                     "optimal_cost": dispatch_cost(inst, opt),
                     "optimal_dispatch": opt},
        })
    return rows


def load_environment(num_examples: int = DEFAULT_NUM_EXAMPLES,
                     seed_offset: int = 0, **kwargs):
    """seed_offset enables disjoint train/eval datasets:
    load_environment(1000) + load_environment(200, seed_offset=1000)."""
    import verifiers as vf
    from datasets import Dataset

    dataset = Dataset.from_list(build_dataset(num_examples, seed_offset=seed_offset))

    def _text(completion):
        # Handles raw strings, dict messages, and Pydantic message models
        # (verifiers >= 0.1.14 passes Pydantic objects — an isinstance(m, dict)
        # filter silently drops them and zeroes all rewards).
        if isinstance(completion, str):
            return completion
        parts = []
        for m in completion:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for p in content:
                    t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return " ".join(parts)

    def fmt(completion, info, **kw):
        return reward_format(_text(completion), info["instance"])

    def bal(completion, info, **kw):
        return reward_power_balance(_text(completion), info["instance"])

    def lim(completion, info, **kw):
        return reward_limits(_text(completion), info["instance"])

    def cost(completion, info, **kw):
        return reward_cost(_text(completion), info["instance"],
                           optimal_cost=info["optimal_cost"])

    rubric = vf.Rubric(funcs=[fmt, bal, lim, cost],
                       weights=list(REWARD_WEIGHTS.values()))
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
