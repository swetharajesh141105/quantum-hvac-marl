"""
day_ahead_planner.py — Change 2: make QUBO matter for ONE building.

Takes the agents' preferred 24h ON/OFF plan and re-plans it with a QUBO that is
tariff- and carbon-aware: it can move ON-hours out of the expensive afternoon
peak into cheaper hours (pre-cooling) while the shared comfort check in
tou_env.evaluate_schedule tells us whether comfort survived.

QUBO per hour t, binary x_t (1 = AC ON), preferred action p_t:
  c_cost*tariff_t*x_t + c_carbon*carbon_t*x_t     price / carbon of running
  c_dev*(1-2*p_t)*x_t                              stay near what agents want
  c_switch*(x_t - x_t+1)^2                         avoid rapid cycling
  c_count*(sum x - K)^2, K = sum p                 keep total ON-hours ~ same

Weights are TUNED on seeds 100-109 and REPORTED on seeds 0-9 (no overlap).
"""
import itertools, os
import numpy as np
from tou_env import TARIFF, CARBON, HOURS, evaluate_schedule, rollout_policy

TUNE_SEEDS = list(range(100, 110))
TEST_SEEDS = list(range(0, 10))
T_N, C_N = TARIFF / TARIFF.max(), CARBON / CARBON.max()


def build_qubo(preferred, c_cost=0.4, c_carbon=0.1, c_dev=1.0, c_switch=0.3, c_count=0.2):
    """Returns (h, J): linear array and {(i,j): coeff} quadratic dict (BINARY vars)."""
    p = np.asarray(preferred, dtype=float)
    K = p.sum()
    h = c_cost * T_N + c_carbon * C_N + c_dev * (1 - 2 * p) + c_count * (1 - 2 * K)
    J = {}
    for t in range(HOURS - 1):                       # switching
        h[t] += c_switch; h[t + 1] += c_switch
        J[(t, t + 1)] = J.get((t, t + 1), 0) - 2 * c_switch
    for i in range(HOURS):                           # count penalty pairs
        for j in range(i + 1, HOURS):
            J[(i, j)] = J.get((i, j), 0) + 2 * c_count
    return h, J


def solve(h, J, num_reads=100, seed=0):
    """Quantum-inspired simulated annealing (dwave-samplers). Falls back to a
    tiny numpy annealer only if dwave-samplers isn't installed."""
    try:
        import dimod
        from dwave.samplers import SimulatedAnnealingSampler
        bqm = dimod.BinaryQuadraticModel({i: float(v) for i, v in enumerate(h)}, J, 0.0, dimod.BINARY)
        best = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=seed).first.sample
        return np.array([best[t] for t in range(HOURS)])
    except ImportError:
        rng = np.random.default_rng(seed)
        Jm = np.zeros((HOURS, HOURS))
        for (i, j), v in J.items():
            Jm[i, j] = Jm[j, i] = v
        best_x, best_e = None, np.inf
        for _ in range(min(num_reads, 10)):
            x = rng.integers(0, 2, HOURS).astype(float)
            for T in np.geomspace(2.0, 0.01, 100):
                for i in rng.permutation(HOURS):
                    d = (1 - 2 * x[i]) * (h[i] + Jm[i] @ x)
                    if d < 0 or rng.random() < np.exp(-d / T):
                        x[i] = 1 - x[i]
            e = h @ x + 0.5 * x @ Jm @ x
            if e < best_e:
                best_e, best_x = e, x.copy()
        return best_x.astype(int)


def _score(m):                       # lower is better; violations dominate
    return m["cost"] + 0.5 * m["carbon"] + 5.0 * m["violation_hours"]


def plan(preferred, **w):
    return solve(*build_qubo(preferred, **w))


def tune(prefs_fn, verbose=True):
    """Grid search on TUNE_SEEDS only. prefs_fn(seed) -> preferred 24h array."""
    prefs = {s: prefs_fn(s) for s in TUNE_SEEDS}
    best, best_w = np.inf, None
    for c_cost, c_dev, c_count in itertools.product([0.0, 0.2, 0.4, 0.8], [0.5, 1.0, 2.0],
                                                    [0.0, 0.1, 0.3, 0.6]):
        w = dict(c_cost=c_cost, c_dev=c_dev, c_count=c_count)
        sc = np.mean([_score(evaluate_schedule(plan(prefs[s], **w), s)) for s in TUNE_SEEDS])
        if sc < best:
            best, best_w = sc, w
    if verbose:
        print(f"Tuned on seeds {TUNE_SEEDS[0]}-{TUNE_SEEDS[-1]}: {best_w}  (score {best:.2f})")
    return best_w


def report(prefs_fn, label):
    w = tune(prefs_fn)
    rows = {"agents only": [], "agents + QUBO planner": []}
    for s in TEST_SEEDS:
        p = prefs_fn(s)
        rows["agents only"].append(evaluate_schedule(p, s))
        rows["agents + QUBO planner"].append(evaluate_schedule(plan(p, **w), s))
    print(f"\nTEST seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}  (preferences from: {label})")
    print(f"{'':24s}{'cost':>8s}{'carbon':>8s}{'ON hrs':>8s}{'violations':>12s}")
    for k, v in rows.items():
        f = lambda key: np.mean([m[key] for m in v])
        print(f"{k:24s}{f('cost'):8.2f}{f('carbon'):8.2f}{f('on_hours'):8.1f}{f('violation_hours'):12.2f}")
    return w, rows


def get_prefs_fn():
    """Agent preferences if tariff-aware agents are trained, else a labelled fallback."""
    if all(os.path.exists(f"{o}_tou.zip") for o in ["energy", "comfort", "carbon"]):
        from tou_agents import load_tou_agent, joint_action
        agents = {o: load_tou_agent(o) for o in ["energy", "comfort", "carbon"]}
        return (lambda s: rollout_policy(lambda ob: joint_action(ob, agents), s)[0]), "tariff-aware MARL agents"
    print("tou agents not trained yet (run tou_agents.py) -> using thermostat-rule preferences")
    from tradeoff_sweep import thermostat_policy
    return (lambda s: rollout_policy(thermostat_policy(24.0), s)[0]), "thermostat rule (fallback)"


if __name__ == "__main__":
    fn, label = get_prefs_fn()
    report(fn, label)
