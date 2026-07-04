import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from economic_dispatch import (
    generate_instance, merit_order_solve, linprog_solve, dispatch_cost,
    reward_format, reward_power_balance, reward_limits, reward_cost,
)

N = int(os.environ.get("N_INSTANCES", "200"))

mismatches = 0
for i in range(N):
    inst = generate_instance(i)
    mo = dispatch_cost(inst, merit_order_solve(inst))
    lp_sol = linprog_solve(inst)
    assert lp_sol is not None, f"seed {i} infeasible"
    lp = dispatch_cost(inst, lp_sol)
    if abs(mo - lp) > max(1.0, 1e-3 * abs(lp)):
        mismatches += 1
        print(f"MISMATCH seed {i}: merit ${mo:.2f} vs linprog ${lp:.2f}")
print(f"cross-check: {N} instances | mismatches={mismatches}")
assert mismatches == 0

inst = generate_instance(0)
opt = merit_order_solve(inst)
oc = dispatch_cost(inst, opt)
ans = lambda d: json.dumps({k: round(v, 3) for k, v in d.items()})

a = ans(opt)
assert reward_format(a, inst) == 1.0 and reward_cost(a, inst, oc) > 0.999
print("gate 1 (optimal): PASS")

cheat = {u["name"]: 0.0 for u in inst["units"]}
cheapest = min(inst["units"], key=lambda u: u["cost"])["name"]
cheat[cheapest] = inst["load_mw"]
a = ans(cheat)
assert reward_cost(a, inst, oc) == 0.0, "cheat earned cost reward — gate broken"
print("gate 2 (all-load-on-cheapest cheat): PASS — cost reward 0.0")

assert reward_format("forty-two", inst) == 0.0
print("gate 3 (garbage): PASS")

# feasible but pricey: proportional split within bounds via linprog on inverted costs
inst_inv = json.loads(json.dumps(inst))
mx = max(u["cost"] for u in inst_inv["units"])
for u in inst_inv["units"]:
    u["cost"] = mx + 1 - u["cost"]
sub = merit_order_solve(inst_inv)
a = ans(sub)
r = reward_cost(a, inst, oc)
assert 0.0 < r < 1.0, f"expected partial credit, got {r}"
print(f"gate 4 (feasible suboptimal): PASS — partial credit {r:.3f}")

# NaN/Infinity attack: json accepts these literals; NaN defeats comparison-based
# checks (every comparison is False). Must score 0 on every reward.
names = [u["name"] for u in inst["units"]]
for bad in ("NaN", "Infinity", "-Infinity"):
    a = "{" + ", ".join(f'"{n}": {bad}' for n in names) + "}"
    assert reward_format(a, inst) == 0.0, f"{bad}: format gate broken"
    assert reward_power_balance(a, inst) == 0.0, f"{bad}: balance gate broken"
    assert reward_limits(a, inst) == 0.0, f"{bad}: limits gate broken"
    assert reward_cost(a, inst, oc) == 0.0, f"{bad}: cost gate broken"
print("gate 5 (NaN/Infinity attack): PASS — all rewards 0.0")

# tolerance-rent attack: shave 0.499 MW off the priciest dispatched unit —
# stays inside the 0.5 MW feasibility tolerance but costs LESS than the
# optimum. A bare min(1, opt/actual) clamp would score this a perfect 1.0.
cheat6 = dict(opt)
victim = max((u for u in inst["units"] if cheat6[u["name"]] > u["p_min"] + 0.5),
             key=lambda u: u["cost"], default=None)
assert victim is not None
cheat6[victim["name"]] -= 0.499
a = ans(cheat6)
r_honest = reward_cost(ans(opt), inst, oc)
r_cheat = reward_cost(a, inst, oc)
assert r_cheat < 0.999 and r_cheat < r_honest, (
    f"tolerance-rent attack scored {r_cheat} vs honest {r_honest} — guard broken")
print(f"gate 6 (tolerance-rent attack): PASS — {r_cheat:.4f} < honest {r_honest:.4f}")

print("ALL VALIDATION PASSED")
